window.initCg = async function (sel, slug, {clickedId, clickedIdCb, isModal, isGridsnap, pruningThreshold} = {}){
  var data = await util.getFile(`./graph_data/${slug}.json`)

  // Client-side edge cap: keep the top-P% of links by |weight| before
  // rendering. Raw graphs can carry 1M+ edges; trimming to ~10% already
  // preserves almost all visible mass and is ~10x cheaper for the browser.
  // Override via URL: ?maxLinksPct=25 (0 = no edges, 100 = keep all).
  var maxLinksPct = parseInt(new URLSearchParams(location.search).get('maxLinksPct') ?? '10', 10)
  if (Number.isFinite(maxLinksPct) && maxLinksPct >= 0 && maxLinksPct < 100 && data.links && data.links.length > 0) {
    var keepN = Math.max(0, Math.round(data.links.length * maxLinksPct / 100))
    if (keepN < data.links.length) {
      data.links.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
      data.links = data.links.slice(0, keepN)
    }
  }
  
  var visState = {
    pinnedIds: [],
    hiddenIds: [],
    hoveredId: null,
    hoveredNodeId: null,
    hoveredCtxIdx: null,
    clickedId: null, 
    clickedCtxIdx: null,
    linkType: 'either',
    isShowAllLinks: '',
    isSyncEnabled: '',
    subgraph: null,
    isEditMode: 1,
    isHideLayer: data.metadata.scan == util.scanSlugToName.h35 || data.metadata.scan == util.scanSlugToName.moc,
    graphSchemaVersion: data.metadata?.schema_version || 0,
    sg_pos: '',
    // isModal is passed in from options; shorthand propagates the option
    // value into visState instead of hardcoding `true`, which was forcing
    // gridsnap into fill-container mode even in the embedded pertok layout
    // (where the host container has no fixed height → grid collapses).
    isModal,
    isGridsnap,
    slug: slug, // Store slug for localStorage keys
    pruningThreshold: pruningThreshold,
    ...(data.qParams ? Object.fromEntries(Object.entries(data.qParams).filter(([_, v]) => v !== null)) : {})
  }
  
  // Get pinnedIds from URL parameters if available (prioritized over localStorage)
  var urlPinnedIds = util.params.get('pinnedIds');
  var urlHiddenIds = util.params.get('hiddenIds');
  console.log(urlPinnedIds, urlHiddenIds)
  console.log(visState.pinnedIds, visState.hiddenIds)
  if (urlPinnedIds) visState.pinnedIds = urlPinnedIds
  if (urlHiddenIds) visState.hiddenIds = urlHiddenIds

  let urlSupernodes = [];
  try {
    const supernodesParam = util.params.get('supernodes');
    if (supernodesParam) {
      urlSupernodes = JSON.parse(supernodesParam);
    }
  } catch (e) {
    console.error('Error parsing supernodes from URL:', e);
  }
  if (urlSupernodes) visState.supernodes = urlSupernodes

  if (visState.clickedId?.includes('supernode')) delete visState.clickedId
  if (clickedId && clickedId != 'null' && !clickedId.includes('supernode-')) visState.clickedId = clickedId
  if (!visState.clickedId || visState.clickedId == 'null' || visState.clickedId == 'undefined') visState.clickedId = data.nodes.find(d => d.isLogit)?.nodeId
  if (visState.pinnedIds?.replace) visState.pinnedIds = visState.pinnedIds.split(',')
  if (visState.hiddenIds?.replace) visState.hiddenIds = visState.hiddenIds.split(',')


  // Load clerps from URL params
  const clerpsParam = util.params.get('clerps') || data.qParams.clerps;
  if (clerpsParam) {
    const clerps = JSON.parse(clerpsParam);
    visState.clerps = new Map(clerps);
  }
  data = await utilCg.formatData(data, visState)
  
  var renderAll = util.initRenderAll(['hClerpUpdate', 'clickedId', 'hiddenIds', 'pinnedIds', 'linkType', 'isShowAllLinks', 'features', 'isSyncEnabled', 'shouldSortByWeight', 'hoveredId'])

  function colorNodes() {
    data.nodes.forEach(d => d.nodeColor = '#fff')
  }
  colorNodes()

  // global link color —  the color scale skips #fff so links are visible
  // TODO: weight by input sum instead
  function colorLinks() {
    var absMax = d3.max(data.links, d => d.absWeight)
    var _linearAbsScale = d3.scaleLinear().domain([-absMax, absMax])
    var _linearPctScale = d3.scaleLinear().domain([-.4, .4])
    var _linearTScale = d3.scaleLinear().domain([0, .5, .5, 1]).range([0, .5 - .001, .5 + .001, 1])

    var widthScale = d3.scaleSqrt().domain([0, 1]).range([.00001, 3])

    utilCg.pctInputColorFn = d => d3.interpolatePRGn(_linearTScale(_linearPctScale(d)))

    data.links.forEach(d => {
      // d.color = d3.interpolatePRGn(_linearTScale(_linearAbsScale(d.weight)))
      d.strokeWidth = widthScale(Math.abs(d.pctInput))
      d.pctInputColor = utilCg.pctInputColorFn(d.pctInput)
      d.color = d3.interpolatePRGn(_linearTScale(_linearPctScale(d.pctInput)))
    })
  }
  colorLinks()

  renderAll.hClerpUpdate.fns.push(params => utilCg.hClerpUpdateFn(params, data))

  renderAll.hoveredId.fns.push(() => {
    // use hovered node if possible, otherwise use last occurence of feature
    var targetCtxIdx = visState.hoveredCtxIdx ?? 999
    var hoveredNodes = data.nodes.filter(n => n.featureId == visState.hoveredId)
    var node = d3.sort(hoveredNodes, d => Math.abs(d.ctx_idx - targetCtxIdx))[0]
    visState.hoveredNodeId = node?.nodeId
  })

  // set tmpClickedLink w/ strength of all the links connected the clickedNode
  renderAll.clickedId.fns.push(() => {
    clickedIdCb?.(visState.clickedId)

    var node = data.nodes.idToNode[visState.clickedId]
    if (!node){
      // for a clicked supernode, sum over memberNode links to make tmpClickedLink
      if (visState.clickedId?.startsWith('supernode-')) {
        node = {
          nodeId: visState.clickedId,
          memberNodes: visState.subgraph.supernodes[+visState.clickedId.split('-')[1]]
            .slice(1)
            .map(id => data.nodes.idToNode[id])
        }
        node.memberSet = new Set(node.memberNodes.map(d => d.nodeId))

        function combineLinks(links, isSrc) {
          return d3.nestBy(links, d => isSrc ? d.sourceNode.nodeId : d.targetNode.nodeId)
            .map(links => ({
              source: isSrc ? links[0].sourceNode.nodeId : visState.clickedId,
              target: isSrc ? visState.clickedId : links[0].targetNode.nodeId,
              sourceNode: isSrc ? links[0].sourceNode : node,
              targetNode: isSrc ? node : links[0].targetNode,
              weight: d3.sum(links, d => d.weight),
              absWeight: Math.abs(d3.sum(links, d => d.weight))
            }))
        }

        node.sourceLinks = combineLinks(node.memberNodes.flatMap(d => d.sourceLinks), true)
        node.targetLinks = combineLinks(node.memberNodes.flatMap(d => d.targetLinks), false)
      } else {
        return data.nodes.forEach(d => {
          d.tmpClickedLink = null
          d.tmpClickedSourceLink = null
          d.tmpClickedTargetLink = null
        })
      }
    }

    var connectedLinks = [...node.sourceLinks, ...node.targetLinks]
    var max = d3.max(connectedLinks, d => d.absWeight)
    var colorScale = d3.scaleSequential(d3.interpolatePRGn).domain([-max*1.1, max*1.1])

    // allowing supernode links means each node can have a both tmpClickedSourceLink and tmpClickedTargetLink
    // currently we render bidirectional links where possible, falling back to the target side links otherwises
    var nodeIdToSourceLink = {}
    var nodeIdToTargetLink = {}
    var featureIdToLink = {}
    connectedLinks.forEach(link => {
      if (link.sourceNode === node) {
        nodeIdToTargetLink[link.targetNode.nodeId] = link
        featureIdToLink[link.targetNode.featureId] = link
        link.tmpClickedCtxOffset = link.targetNode.ctx_idx - node.ctx_idx
      }
      if (link.targetNode === node) {
        nodeIdToSourceLink[link.sourceNode.nodeId] = link
        featureIdToLink[link.sourceNode.featureId] = link
        link.tmpClickedCtxOffset = link.sourceNode.ctx_idx - node.ctx_idx
      }
      // link.tmpColor = colorScale(link.pctInput)
      link.tmpColor = link.pctInputColor
    })

    data.nodes.forEach(d => {
      var link = nodeIdToSourceLink[d.nodeId] || nodeIdToTargetLink[d.nodeId]
      d.tmpClickedLink = link
      d.tmpClickedSourceLink = nodeIdToSourceLink[d.nodeId]
      d.tmpClickedTargetLink = nodeIdToTargetLink[d.nodeId]
    })

    data.features.forEach(d => {
      var link = featureIdToLink[d.featureId]
      d.tmpClickedLink = link
    })
  })

  function initGridsnap() {
    var gridData = [
      // {cur: {x: 0, y: 0,  w: 6, h: .5}, class: 'button-container'},
      // Link graph spans the full 14-column width by default. Subgraph goes
      // below it (also full width). Node-connections + feature-detail live
      // further down, split into two columns. pertok.html's tab toggle can
      // hide whichever set of panels isn't active.
      {cur: {x: 0, y: 0,  w: 14, h: 14}, class: 'link-graph', resizeFn: makeResizeFn(initCgLinkGraph)},
      {cur: {x: 0, y: 16, w: 14, h: 16}, class: 'subgraph'},
      {cur: {x: 0, y: 32, w: 8,  h: 14}, class: 'node-connections'},
      {cur: {x: 8, y: 32, w: 6,  h: 20}, class: 'feature-detail'},
      // {cur: {x: 0, y: 18, w: 6, h: 7}, class: 'clerp-list'},
      // {cur: {x: 6, y: 30, w: 4, h: 7}, class: 'feature-scatter'},
      // {cur: {x: 0, y: 30, w: 3, h: 8}, class: 'metadata'},
     ].filter(d => d)

    var initFns = [
      initCgButtonContainer,   
      initCgSubgraph,
      initCgLinkGraph,
      initCgNodeConnections, 
      initCgFeatureDetail, 
      // initCgClerpList, 
      // initCgFeatureScatter, 
    ].filter(d => d)
    
    var gridsnapSel = sel.html('').append('div.gridsnap.cg')
      .classed('is-edit-mode', visState.isGridsnap)
    if (visState.isModal) gridsnapSel.st({width: '100%', height: '100%'})

    
    window.initGridsnap({
      gridData,
      gridSizeY: 60,
      pad: 10,
      sel: gridsnapSel,
      isFullScreenY: false,
      isFillContainer: visState.isModal,
      // serializedGrid intentionally omitted: each graph JSON ships with a
      // stale gridsnap layout (h:8 for link-graph) that was baked at dump
      // time. Honoring it here collapses the panel on load, overriding the
      // tall default above. Drop it so gridData always wins.
    })

    initFns.forEach(fn => fn({visState, renderAll, data, cgSel: sel}))

    function makeResizeFn(fn){
      return () => {
        fn({visState, renderAll, data, cgSel: sel.select('.gridsnap.cg')})
        Object.values(renderAll).forEach(d => d())
      }
    }
  }

  initGridsnap()
  renderAll.hClerpUpdate()
  renderAll.isShowAllLinks()
  renderAll.linkType()
  renderAll.clickedId()
  renderAll.pinnedIds()
  renderAll.features()
  renderAll.isSyncEnabled()
  renderAll.hoveredId()
}

window.init?.()
