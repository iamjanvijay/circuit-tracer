import argparse
import json
import logging
import os
import time
import warnings


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="CLI for attribution, graph file creation, and server hosting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Attribution subcommand
    attr_parser = subparsers.add_parser("attribute", help="Run attribution analysis on a prompt")

    # Arguments from attribute_batch.py
    attr_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=("Model architecture to use for attribution. Can be inferred from transcoder config."),
    )
    attr_parser.add_argument(
        "-t",
        "--transcoder_set",
        required=True,
        help=(
            "HuggingFace repository ID containing transcoders "
            "(e.g. username/repo-name, username/repo-name@revision)."
        ),
    )
    attr_parser.add_argument(
        "-p", "--prompt",
        required=False,
        help=(
            "Input prompt text to analyze. Required unless --prompt_token_ids "
            "is provided."
        ),
    )
    attr_parser.add_argument(
        "--prompt_token_ids",
        type=str,
        help=(
            "Pre-tokenized full prompt as a JSON list of ints (or a @file "
            "path). When supplied we skip string tokenization entirely and "
            "use these ids verbatim. --prompt is used only for logging."
        ),
    )
    attr_parser.add_argument(
        "--prefix_token_len",
        type=int,
        help=(
            "Number of prefix tokens when combined with --per_token. Per-token "
            "attribution starts at this index. Equivalent to tokenizing "
            "--prefix_prompt and measuring its length, but lets callers pass "
            "the split point derived from token_regions metadata directly."
        ),
    )
    attr_parser.add_argument(
        "-o",
        "--graph_output_path",
        help=(
            "Path where to save the attribution graph (.pt file). Required if not "
            "creating graph files."
        ),
    )
    attr_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
        default="float32",
        help="Data type for model weights (default: float32).",
    )
    attr_parser.add_argument(
        "--max_n_logits", type=int, default=10, help="Maximum number of logit nodes."
    )
    attr_parser.add_argument(
        "--desired_logit_prob",
        type=float,
        default=0.95,
        help="Cumulative probability threshold for top logits.",
    )
    attr_parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for backward passes."
    )
    attr_parser.add_argument(
        "--offload",
        choices=["cpu", "disk", None],
        default=None,
        help="Offload model parameters to save memory.",
    )
    attr_parser.add_argument(
        "--max_feature_nodes",
        type=int,
        default=7500,
        help="Maximum number of feature nodes.",
    )
    attr_parser.add_argument("--verbose", action="store_true", help="Display progress information.")
    attr_parser.add_argument(
        "--lazy-encoder",
        action="store_true",
        help="Enable lazy loading for encoder weights to save memory.",
    )
    attr_parser.add_argument(
        "--lazy-decoder",
        action="store_true",
        default=True,
        help="Enable lazy loading for decoder weights to save memory (default: True).",
    )
    attr_parser.add_argument(
        "--backend",
        type=str,
        choices=["transformerlens", "nnsight"],
        default="transformerlens",
        help="Backend to use for the replacement model (default: transformerlens).",
    )

    # Arguments for graph creation
    attr_parser.add_argument(
        "--slug",
        type=str,
        help=(
            "Slug for the model metadata (used for graph files). Required if creating "
            "graph files or starting server."
        ),
    )
    attr_parser.add_argument(
        "--graph_file_dir",
        type=str,
        help=(
            "Path to save the output JSON graph files, and also used as data dir for "
            "server. Required if creating graph files or starting server."
        ),
    )
    attr_parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.8,
        help="Node threshold for pruning graph files.",
    )
    attr_parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.98,
        help="Edge threshold for pruning graph files.",
    )
    attr_parser.add_argument(
        "--per_token",
        action="store_true",
        help=(
            "Build one attribution graph per prompt-token position: for each real "
            "token position i (skipping the BOS-only prefix), the target is the "
            "ground-truth token t_i conditioned on its prefix [BOS, t_1..t_{i-1}]. "
            "Requires --slug and --graph_file_dir; incompatible with --graph_output_path."
        ),
    )
    attr_parser.add_argument(
        "--prefix_prompt",
        type=str,
        default=None,
        help=(
            "DEPRECATED — rejected at runtime. Use --prompt_token_ids + "
            "--prefix_token_len with --per_token instead."
        ),
    )
    attr_parser.add_argument(
        "--abstractions",
        nargs="+",
        default=["none"],
        help=(
            "Graph abstractions to emit alongside the default graph. 'none' is "
            "always included. Extra abstractions are written as "
            "{slug}__{abstraction}.json. Pertok UI exposes these via a dropdown. "
            "Supported: none, token_level."
        ),
    )

    # Server arguments
    attr_parser.add_argument(
        "--server",
        action="store_true",
        help="Start a local server to visualize graphs after processing.",
    )
    attr_parser.add_argument("--port", type=int, default=8041, help="Port for the local server.")
    attr_parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Path to the directory containing feature files for local server, if using local transcoders (default: None)",
    )

    # Start-server subcommand
    server_parser = subparsers.add_parser(
        "start-server", help="Start a local server to visualize existing graphs"
    )
    server_parser.add_argument(
        "--graph_file_dir",
        type=str,
        required=True,
        help="Path to the directory containing graph JSON files.",
    )
    server_parser.add_argument(
        "--features_dir",
        type=str,
        default=None,
        help="Path to the directory containing feature files for local server, if using local transcoders (default: None)",
    )
    server_parser.add_argument("--port", type=int, default=8041, help="Port for the local server.")

    args = parser.parse_args()

    if args.command == "attribute":
        run_attribution(args, attr_parser)
    if args.command == "start-server" or args.server:
        run_server(args)


def run_attribution(args, parser):
    # Check if one of slug/graph_file_dir is provided but not the other
    if bool(args.slug) != bool(args.graph_file_dir):
        which_one = "slug" if args.slug else "graph_file_dir"
        missing_one = "graph_file_dir" if args.slug else "slug"
        warnings.warn(
            (
                f"You provided --{which_one} but not --{missing_one}. Both are required "
                "for creating graph files."
            ),
            UserWarning,
        )

    # Determine if we're creating graph files
    create_graph_files_enabled = args.slug is not None and args.graph_file_dir is not None

    # Validate arguments
    if args.server and (not args.slug or not args.graph_file_dir):
        parser.error("Both --slug and --graph_file_dir are required when using --server")

    if not create_graph_files_enabled and not args.graph_output_path:
        parser.error(
            "--graph_output_path is required when not creating graph files "
            "(--slug and --graph_file_dir)"
        )

    if args.per_token:
        if not create_graph_files_enabled:
            parser.error(
                "--per_token requires both --slug and --graph_file_dir "
                "(one JSON is written per prompt position)."
            )
        if args.graph_output_path:
            parser.error(
                "--per_token is incompatible with --graph_output_path "
                "(per-token mode writes multiple graphs, not a single .pt file)."
            )
    # --prefix_prompt is retired: per_token takes --prompt_token_ids +
    # --prefix_token_len instead. Reject it clearly if anyone still passes it.
    if args.prefix_prompt is not None:
        parser.error(
            "--prefix_prompt is no longer supported. Use --prompt_token_ids "
            "and --prefix_token_len with --per_token instead."
        )
    if args.prefix_token_len is not None and not args.per_token:
        parser.error("--prefix_token_len is only valid together with --per_token.")
    if args.prompt is None and args.prompt_token_ids is None:
        parser.error("Either --prompt or --prompt_token_ids is required.")

    if args.abstractions:
        from circuit_tracer.utils import abstractions as abstractions_mod

        valid = set(abstractions_mod.available())
        bad = [a for a in args.abstractions if a not in valid]
        if bad:
            parser.error(
                f"unknown --abstractions values: {bad}; available: {sorted(valid)}"
            )

    # Ensure graph output directory exists if needed
    if create_graph_files_enabled:
        os.makedirs(args.graph_file_dir, exist_ok=True)

    import torch

    dtype = args.dtype
    # Convert short dtype string to long dtype string
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    if dtype in dtype_mapping:
        dtype = dtype_mapping[dtype]
    dtype = getattr(torch, dtype)

    # Run attribution
    logging.info(f"Generating attribution graph for model: {args.model}")
    logging.info(f"Loading model with dtype: {dtype}")
    if args.prompt is not None:
        logging.info(f'Input prompt: "{args.prompt}"')
    if args.prompt_token_ids is not None:
        logging.info("Using pre-tokenized input via --prompt_token_ids")
    if args.graph_output_path:
        logging.info(f"Output will be saved to: {args.graph_output_path}")
    logging.info(
        f"Including logits with cumulative probability >= {args.desired_logit_prob} "
        f"(max {args.max_n_logits})"
    )
    logging.info(f"Using batch size of {args.batch_size} for backward passes")

    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.utils.create_graph_files import create_graph_files
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    transcoder, config = load_transcoder_from_hub(
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=args.lazy_encoder,
        lazy_decoder=args.lazy_decoder,
    )
    args.model = args.model or config.get("model_name", None)
    if not args.model:
        parser.error("--model must be specified when not provided in transcoder config")

    model_instance = ReplacementModel.from_pretrained_and_transcoders(
        args.model, transcoder, dtype=dtype, backend=args.backend
    )

    if args.per_token:
        # Per-token mode accepts ONLY pre-tokenized input. Callers are
        # responsible for tokenizing their prompt once (e.g. via
        # build_prompts.py) and passing both the full token ids and the
        # number of prefix tokens.
        if args.prompt_token_ids is None:
            parser.error(
                "--per_token requires --prompt_token_ids (string --prompt / "
                "--prefix_prompt paths were removed). Pass a JSON list of "
                "ints directly or via @file."
            )
        if args.prefix_token_len is None:
            parser.error(
                "--per_token requires --prefix_token_len — the index of the "
                "first generated (target) token in --prompt_token_ids."
            )
        raw = args.prompt_token_ids
        if raw.startswith("@"):
            with open(raw[1:], "r") as fh:
                raw = fh.read()
        try:
            token_id_list = json.loads(raw)
        except json.JSONDecodeError as exc:
            parser.error(f"--prompt_token_ids is not valid JSON: {exc}")
        if not isinstance(token_id_list, list) or not all(
            isinstance(x, int) for x in token_id_list
        ):
            parser.error("--prompt_token_ids must be a JSON list of ints")
        input_ids = torch.tensor(token_id_list, dtype=torch.long)
        n = int(input_ids.shape[0])
        if n < 2:
            parser.error(
                f"--per_token needs at least 1 non-BOS token "
                f"(got tokenized length {n})."
            )
        m = int(args.prefix_token_len)
        if m < 1 or m >= n:
            parser.error(
                f"--prefix_token_len={m} out of range for a prompt of {n} tokens "
                f"(must be in [1, {n - 1}])."
            )
        start_i = m
        logging.info(
            f"--prefix_token_len covers {m} tokens; per-token attribution starts "
            f"at position {start_i}"
        )
        logging.info(
            f"Running per-token attribution over {n - start_i} positions "
            f"(i={n - 1}..{start_i}, reverse order: last target first)"
        )
        for i in range(n - 1, start_i - 1, -1):
            prefix_ids = input_ids[:i]
            target_id = input_ids[i : i + 1]
            target_str = model_instance.tokenizer.decode(target_id.tolist())
            logging.info(
                f"[per_token] position {i}/{n - 1}: target id={int(target_id)} ({target_str!r})"
            )
            graph = attribute(
                prompt=prefix_ids,
                model=model_instance,  # type:ignore
                attribution_targets=target_id,
                max_n_logits=args.max_n_logits,
                desired_logit_prob=args.desired_logit_prob,
                batch_size=args.batch_size,
                verbose=args.verbose,
                offload=args.offload,
                max_feature_nodes=args.max_feature_nodes,
            )
            per_slug = f"{args.slug}-pos{i:03d}"
            if os.environ.get("CT_WRITE_PT", "1") != "0":
                pt_path = os.path.join(args.graph_file_dir, f"{per_slug}.pt")
                logging.info(f"Saving graph to {pt_path}")
                graph.to_pt(pt_path)
            logging.info(f"Creating graph files with slug: {per_slug}")
            create_graph_files(
                graph_or_path=graph,
                slug=per_slug,
                scan=None,
                output_path=args.graph_file_dir,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                abstractions=args.abstractions,
            )
        logging.info(f"Per-token graph files (.pt + JSON) written to {args.graph_file_dir}")
    else:
        logging.info("Running attribution...")
        graph = attribute(
            prompt=args.prompt,
            model=model_instance,  # type:ignore
            max_n_logits=args.max_n_logits,
            desired_logit_prob=args.desired_logit_prob,
            batch_size=args.batch_size,
            verbose=args.verbose,
            offload=args.offload,
            max_feature_nodes=args.max_feature_nodes,
        )

        # Save to file if output path specified
        if args.graph_output_path:
            logging.info(f"Saving graph to {args.graph_output_path}")
            graph.to_pt(args.graph_output_path)

        # Create graph files if both slug and graph_file_dir are provided
        if create_graph_files_enabled:
            logging.info(f"Creating graph files with slug: {args.slug}")
            create_graph_files(
                graph_or_path=graph,  # Use the graph object directly
                slug=args.slug,
                scan=None,  # No scan argument needed
                output_path=args.graph_file_dir,
                node_threshold=args.node_threshold,
                edge_threshold=args.edge_threshold,
                abstractions=args.abstractions,
            )
            logging.info(f"Graph JSON files written to {args.graph_file_dir}")


def run_server(args):
    from circuit_tracer.frontend.local_server import serve

    logging.info(f"Starting server on port {args.port}...")
    logging.info(f"Serving data from: {os.path.abspath(args.graph_file_dir)}")
    if args.features_dir:
        if not os.path.isdir(args.features_dir):
            raise ValueError(f"features_dir does not exist: {args.features_dir}")
        logging.info(f"Using features directory: {os.path.abspath(args.features_dir)}")
    server = serve(data_dir=args.graph_file_dir, port=args.port, features_dir=args.features_dir)
    try:
        logging.info("Press Ctrl+C to stop the server.")
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        logging.info("Stopping server...")
        server.stop()


if __name__ == "__main__":
    main()
