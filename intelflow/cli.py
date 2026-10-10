"""
Command-Line Interface for IntelFlowLM (inpu / flm-intel).
Provides 100% FastFlowLM feature parity: run, serve, pull, list, remove, check,
validate, port, benchmark/bench, top, doctor, and mcp.
"""

import os
import sys
import time
import json
import re
from typing import Optional, List, Dict, Any
from pathlib import Path
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.panel import Panel

from intelflow.config import DEFAULT_MODEL, DEFAULT_HOST, DEFAULT_PORT, MODEL_REGISTRY, resolve_model_tag
from intelflow.hub import ModelHub
from intelflow.doctor import check_kernel_driver, check_openvino_npu, run_doctor

app = typer.Typer(
    name="inpu",
    help="⚡ IntelFlowLM: Ultra-efficient LLM runtime for Intel Core Ultra NPUs (FastFlowLM counterpart)",
    add_completion=False,
)
console = Console()


@app.command()
def doctor():
    """Verify Intel NPU hardware, Linux DRM driver, and Level Zero runtime."""
    success = run_doctor()
    if not success:
        raise typer.Exit(code=1)


@app.command()
def validate(
    json_output: bool = typer.Option(False, "--json", help="Output validation results as a JSON object"),
):
    """Verify that your hardware meets necessary requirements to run IntelFlowLM."""
    drm_info = check_kernel_driver()
    npu_info = check_openvino_npu()

    all_ready = drm_info["status"] and npu_info["status"]

    if json_output:
        res = {
            "valid": all_ready,
            "device": npu_info.get("full_name") or "Intel NPU",
            "architecture": npu_info.get("arch") or "Unknown",
            "drm_node": drm_info.get("path") or "/dev/accel/accel0",
            "drm_status": "READY" if drm_info["status"] else "FAILED",
            "drm_details": drm_info["message"],
            "openvino_status": "OPTIMAL" if npu_info["status"] else "MISSING",
            "capabilities": npu_info.get("caps") or [],
        }
        console.print(json.dumps(res, indent=2))
        return

    success = run_doctor()
    if not success:
        raise typer.Exit(code=1)


@app.command()
def port():
    """Show current IntelFlowLM server port (default 52625)."""
    console.print(str(DEFAULT_PORT))


@app.command()
def pull(
    model_tag: str = typer.Argument(DEFAULT_MODEL, help="Model tag or Hugging Face repo ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-download of model files"),
):
    """Download an optimized OpenVINO INT4 model from Hugging Face."""
    try:
        ModelHub.pull(model_tag, force=force)
    except Exception as e:
        console.print(f"[bold red]Error pulling model:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="remove")
def remove_model(
    model_tag: str = typer.Argument(..., help="Model tag or repo ID to remove from disk"),
):
    """Delete a model from local storage."""
    success = ModelHub.remove(model_tag)
    if success:
        console.print(f"[bold green]✔ Removed model '{model_tag}' from local storage.[/bold green]")
    else:
        console.print(f"[yellow]Model '{model_tag}' was not found in local storage.[/yellow]")


@app.command(name="rm", hidden=True)
def rm_model(model_tag: str = typer.Argument(...)):
    """Alias for remove."""
    remove_model(model_tag)


@app.command(name="check")
def check_model(
    model_tag: str = typer.Argument(..., help="Model tag or repo ID to verify"),
    json_output: bool = typer.Option(False, "--json", help="Output validation as JSON"),
):
    """Verify file integrity for a downloaded model."""
    res = ModelHub.check(model_tag)
    if json_output:
        console.print(json.dumps(res, indent=2))
        return

    if not res.get("installed"):
        console.print(f"[bold red]❌ Model '{model_tag}' is not downloaded locally.[/bold red]")
        raise typer.Exit(code=1)

    if res.get("valid"):
        console.print(f"[bold green]✔ Model '{model_tag}' is verified and complete![/bold green]")
        console.print(f"[dim]Disk footprint: {res['size_mb']} MB across {res['file_count']} files.[/dim]")
    else:
        console.print(f"[bold red]❌ Model '{model_tag}' has missing or corrupted artifacts.[/bold red]")
        raise typer.Exit(code=1)


@app.command(name="list")
def list_models(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON object"),
    filter_type: str = typer.Option("all", "--filter", help="Filter models: all, installed, not-installed"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal quiet view (no icons/table)"),
):
    """Display supported and downloaded models."""
    installed = ModelHub.list_installed()
    installed_tags = {m["tag"]: m for m in installed}
    catalog = ModelHub.list_catalog()

    if json_output:
        all_models = []
        for tag, meta in catalog.items():
            is_inst = tag in installed_tags
            if filter_type == "installed" and not is_inst:
                continue
            if filter_type == "not-installed" and is_inst:
                continue
            all_models.append({
                "tag": tag,
                "repo_id": meta.get("repo_id"),
                "description": meta.get("description"),
                "type": meta.get("type", "llm"),
                "installed": is_inst,
                "size_mb": installed_tags[tag]["size_mb"] if is_inst else None,
            })
        console.print(json.dumps(all_models, indent=2))
        return

    if quiet:
        for tag, meta in catalog.items():
            is_inst = tag in installed_tags
            if filter_type == "installed" and not is_inst:
                continue
            if filter_type == "not-installed" and is_inst:
                continue
            status = "installed" if is_inst else "available"
            console.print(f"{tag}\t{status}\t{meta.get('repo_id')}")
        return

    # Rich table view
    if filter_type in ("all", "installed"):
        console.print("\n[bold cyan]📦 Locally Cached Models[/bold cyan]\n")
        if not installed:
            console.print("[dim]No models cached yet. Run 'inpu pull <model>' to download one.[/dim]\n")
        else:
            table = Table(expand=True)
            table.add_column("Tag", style="bold green", width=20)
            table.add_column("Repo ID", style="white", width=45)
            table.add_column("Disk Size", style="cyan", width=12)
            table.add_column("Local Path", style="dim")
            for m in installed:
                table.add_row(m["tag"], m["repo_id"], f"{m['size_mb']} MB", m["path"])
            console.print(table)

    if filter_type in ("all", "not-installed"):
        console.print("\n[bold cyan]🌟 Available Curated Catalog[/bold cyan]\n")
        cat_table = Table(expand=True)
        cat_table.add_column("Tag", style="bold yellow", width=20)
        cat_table.add_column("Description", style="white", width=45)
        cat_table.add_column("Status", style="dim", width=14)

        for tag, meta in catalog.items():
            is_inst = tag in installed_tags
            if filter_type == "not-installed" and is_inst:
                continue
            stat_text = "[green]installed[/green]" if is_inst else "[dim]available[/dim]"
            cat_table.add_row(tag, meta.get("description", ""), stat_text)
        console.print(cat_table)
        console.print()


@app.command()
def run(
    model_tag: str = typer.Argument(DEFAULT_MODEL, help="Model tag to run"),
    device: str = typer.Option("NPU", "--device", "-d", help="Execution device: NPU, GPU, CPU"),
    max_tokens: int = typer.Option(512, "--max-tokens", "-m", help="Maximum tokens per response"),
    ctx_len: int = typer.Option(4096, "--ctx-len", "-c", help="Context length limit"),
):
    """Interactive streaming chat REPL on the Intel NPU."""
    from intelflow.engine import NPUEngine

    console.print(f"\n[bold cyan]⚡ Loading {model_tag} on {device.upper()}...[/bold cyan]")
    t0 = time.perf_counter()
    try:
        engine = NPUEngine(model_tag, device=device)
    except Exception as e:
        console.print(f"[bold red]Initialization failed:[/bold red] {e}")
        raise typer.Exit(code=1)

    load_time = time.perf_counter() - t0
    console.print(f"[green]✔ Model ready in {load_time:.2f}s![/green]")
    console.print("[dim]Type your message. Interactive commands: /?, /show, /clear, /history, /status, /verbose, /save, /input, /bye[/dim]\n")

    verbose = False
    temperature = 0.7
    top_p = 0.9
    gen_limit = max_tokens
    chat_history: List[Dict[str, str]] = []

    engine.start_chat()

    while True:
        try:
            user_input = console.input("[bold blue]>>> [/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting session...[/dim]")
            break

        if not user_input:
            continue

        # FastFlowLM Slash Commands
        if user_input in ("/?", "/help"):
            help_panel = (
                "[bold cyan]Available Interactive Commands:[/bold cyan]\n"
                "  [green]/? [/green] or [green]/help[/green]         Show this help reference\n"
                "  [green]/show[/green]                View model info, architecture, and device details\n"
                "  [green]/clear[/green]               Clear conversation memory and KV cache\n"
                "  [green]/history[/green]             Display current session's conversation history\n"
                "  [green]/status[/green]              Display runtime stats and token throughput\n"
                "  [green]/verbose[/green]             Toggle performance telemetry reporting per turn\n"
                "  [green]/save [file][/green]         Save conversation transcript to disk\n"
                "  [green]/load <model>[/green]        Dynamically switch to a different model\n"
                "  [green]/input \"<file>\" <prompt>[/green] Read a plain text file into your prompt\n"
                "  [green]/set <param> <val>[/green]   Configure hyperparameters (e.g. /set gen-lim 256, /set temp 0.2)\n"
                "  [green]/bye[/green] or [green]/exit[/green]       Exit the CLI\n"
            )
            console.print(Panel(help_panel, title="⚡ Commands", border_style="cyan"))
            continue

        elif user_input in ("/bye", "/exit", "/quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        elif user_input == "/show":
            meta = MODEL_REGISTRY.get(engine.model_tag, {})
            console.print(
                f"[bold cyan]Model Tag:[/bold cyan] {engine.model_tag}\n"
                f"[bold cyan]Repository:[/bold cyan] {engine.model_path.name}\n"
                f"[bold cyan]Device Target:[/bold cyan] {engine.device} (Intel Core Ultra)\n"
                f"[bold cyan]Context Length:[/bold cyan] {ctx_len}\n"
                f"[bold cyan]Generate Limit:[/bold cyan] {gen_limit}\n"
                f"[bold cyan]Temperature:[/bold cyan] {temperature} | [bold cyan]Top-P:[/bold cyan] {top_p}\n"
            )
            continue

        elif user_input == "/clear":
            engine.finish_chat()
            engine.start_chat()
            chat_history.clear()
            console.print("[green]✔ Conversation history & KV cache cleared.[/green]\n")
            continue

        elif user_input == "/history":
            if not chat_history:
                console.print("[dim]No messages in current session history yet.[/dim]\n")
            else:
                table = Table(title="Conversation History", expand=True)
                table.add_column("Turn", style="dim", width=6)
                table.add_column("Role", style="bold cyan", width=12)
                table.add_column("Content", style="white")
                for i, item in enumerate(chat_history):
                    role_color = "blue" if item["role"] == "user" else "green"
                    table.add_row(str(i + 1), f"[{role_color}]{item['role']}[/{role_color}]", item["content"])
                console.print(table)
                console.print()
            continue

        elif user_input == "/status":
            m = engine.last_metrics
            if not m:
                console.print("[dim]No inference metrics captured yet. Send a prompt first.[/dim]\n")
            else:
                console.print(
                    f"[bold cyan]Last TTFT:[/bold cyan] {m.get('ttft_ms', 0)} ms\n"
                    f"[bold cyan]Decode Speed:[/bold cyan] {m.get('throughput_tok_s', 0)} tok/s\n"
                    f"[bold cyan]Tokens Generated:[/bold cyan] {m.get('tokens_generated', 0)}\n"
                    f"[bold cyan]Total Time:[/bold cyan] {m.get('total_time_sec', 0)} seconds\n"
                )
            continue

        elif user_input == "/verbose":
            verbose = not verbose
            state = "ON" if verbose else "OFF"
            console.print(f"[yellow]Verbose performance metrics: {state}[/yellow]\n")
            continue

        elif user_input.startswith("/load "):
            new_tag = user_input.split(maxsplit=1)[1].strip()
            console.print(f"[cyan]Switching to model '{new_tag}' on {device}...[/cyan]")
            try:
                engine.finish_chat()
                engine = NPUEngine(new_tag, device=device)
                engine.start_chat()
                chat_history.clear()
                console.print(f"[bold green]✔ Successfully loaded '{new_tag}'![/bold green]\n")
            except Exception as ex:
                console.print(f"[bold red]Failed to load model:[/bold red] {ex}\n")
            continue

        elif user_input.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            filename = parts[1].strip() if len(parts) > 1 else f"conversation_{int(time.time())}.txt"
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    for item in chat_history:
                        f.write(f"[{item['role'].upper()}]\n{item['content']}\n\n")
                console.print(f"[green]✔ Conversation saved to {filename}[/green]\n")
            except Exception as ex:
                console.print(f"[bold red]Failed to save file:[/bold red] {ex}\n")
            continue

        elif user_input.startswith("/set "):
            set_parts = user_input.split(maxsplit=2)
            if len(set_parts) >= 3:
                param, val = set_parts[1].lower(), set_parts[2]
                try:
                    if param in ("gen-lim", "gen_lim", "max-tokens", "max_tokens"):
                        gen_limit = int(val)
                        console.print(f"[green]Set gen-lim to {gen_limit}[/green]\n")
                    elif param in ("temp", "temperature"):
                        temperature = float(val)
                        console.print(f"[green]Set temperature to {temperature}[/green]\n")
                    elif param in ("top-p", "top_p"):
                        top_p = float(val)
                        console.print(f"[green]Set top_p to {top_p}[/green]\n")
                    else:
                        console.print(f"[yellow]Unknown setting '{param}'. Valid: gen-lim, temp, top-p[/yellow]\n")
                except ValueError:
                    console.print(f"[bold red]Invalid value '{val}' for {param}[/bold red]\n")
            continue

        elif user_input.startswith("/input "):
            # Syntax: /input "<file_path>" prompt
            match = re.match(r'^/input\s+"([^"]+)"\s*(.*)$', user_input)
            if match:
                filepath, custom_prompt = match.group(1), match.group(2)
                file_p = Path(filepath)
                if not file_p.exists():
                    console.print(f"[bold red]File not found:[/bold red] {filepath}\n")
                    continue
                try:
                    file_text = file_p.read_text(encoding="utf-8", errors="ignore")
                    user_input = f"{custom_prompt}\n\nContext from {file_p.name}:\n{file_text}"
                except Exception as ex:
                    console.print(f"[bold red]Error reading file:[/bold red] {ex}\n")
                    continue
            else:
                console.print('[yellow]Usage: /input "<file_path>" prompt text[/yellow]\n')
                continue

        # Record user turn
        chat_history.append({"role": "user", "content": user_input})

        # Stream response
        console.print("[bold green]Intel NPU:[/bold green] ", end="")
        full_response = ""
        try:
            for chunk in engine.generate_stream(
                user_input,
                max_new_tokens=gen_limit,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
            ):
                console.print(chunk, end="")
                full_response += chunk
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow][Interrupted][/yellow]")
        except Exception as ex:
            console.print(f"\n[bold red]Generation error:[/bold red] {ex}")

        chat_history.append({"role": "assistant", "content": full_response})

        if verbose and engine.last_metrics:
            m = engine.last_metrics
            console.print(
                f"[dim]Metrics: TTFT: {m.get('ttft_ms')}ms | "
                f"Rate: {m.get('throughput_tok_s')} tok/s | "
                f"Tokens: {m.get('tokens_generated')} | "
                f"Total: {m.get('total_time_sec')}s[/dim]\n"
            )
        else:
            console.print()

    engine.finish_chat()


@app.command()
def serve(
    model_tag: str = typer.Argument(DEFAULT_MODEL, help="Model tag to serve"),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Port to listen on"),
    host: str = typer.Option(DEFAULT_HOST, "--host", "-h", help="Host interface to bind"),
    cors: int = typer.Option(1, "--cors", help="Enable (1) or disable (0) Cross-Origin Resource Sharing"),
):
    """Start the dual OpenAI & Ollama REST server on port 52625."""
    import uvicorn
    from intelflow import server

    server.active_model_name = model_tag
    console.print(f"\n[bold green]⚡ Starting IntelFlowLM server on http://{host}:{port}[/bold green]")
    console.print(f"[cyan]Active Model:[/cyan] {model_tag} [dim](accelerated on Intel NPU)[/dim]")
    console.print(
        "[dim]Endpoints ready:\n"
        "  - OpenAI: /v1/chat/completions, /v1/completions, /v1/models, /v1/embeddings, /v1/version\n"
        "  - Ollama: /api/chat, /api/generate, /api/tags, /api/ps, /api/npu/status, /api/version[/dim]\n"
    )

    uvicorn.run(server.app, host=host, port=port, log_level="info")


@app.command(name="benchmark")
def benchmark(
    model_tag: str = typer.Argument(DEFAULT_MODEL, help="Model tag to benchmark"),
    device: str = typer.Option("NPU", "--device", "-d", help="Execution device: NPU, GPU, CPU"),
    tokens: int = typer.Option(128, "--tokens", "-t", help="Number of tokens to generate"),
    bench_iterations: int = typer.Option(1, "--bench-iterations", help="Number of benchmark iterations"),
    csv_file: Optional[str] = typer.Option(None, "--csv", help="Save benchmark results to CSV"),
):
    """Run standardized prompt latency and token throughput benchmark."""
    from intelflow.engine import NPUEngine

    console.print(f"\n[bold cyan]🏁 Benchmarking {model_tag} on {device.upper()}...[/bold cyan]")
    test_prompt = "Explain why quantum computers use superposition and entanglement in detail."

    engine = NPUEngine(model_tag, device=device)
    console.print("[dim]Warming up graph cache...[/dim]")
    engine.generate("Hello world", max_new_tokens=10)

    console.print(f"[dim]Running benchmark (target {tokens} tokens, {bench_iterations} pass)...[/dim]")
    
    last_res = None
    all_metrics = []
    for it in range(bench_iterations):
        res = engine.generate(test_prompt, max_new_tokens=tokens)
        all_metrics.append(res["metrics"])
        last_res = res

    # Average metrics
    avg_ttft = sum(m.get("ttft_ms", 0.0) for m in all_metrics) / len(all_metrics)
    avg_tps = sum(m.get("throughput_tok_s", 0.0) for m in all_metrics) / len(all_metrics)
    avg_toks = sum(m.get("tokens_generated", 0) for m in all_metrics) / len(all_metrics)
    avg_time = sum(m.get("total_time_sec", 0.0) for m in all_metrics) / len(all_metrics)

    table = Table(title="Benchmark Results", expand=True)
    table.add_column("Metric", style="bold white", width=25)
    table.add_column("Result", style="bold green", width=20)
    table.add_column("Unit", style="dim")

    table.add_row("Device", device.upper(), "-")
    table.add_row("Time to First Token (TTFT)", f"{avg_ttft:.2f}", "ms")
    table.add_row("Decode Throughput", f"{avg_tps:.2f}", "tokens / sec")
    table.add_row("Tokens Generated", f"{int(avg_toks)}", "tokens")
    table.add_row("Total Generation Time", f"{avg_time:.3f}", "seconds")

    console.print(table)
    if last_res:
        console.print(Panel(f"[bold green]Response:[/bold green]\n{last_res['text']}", title="Generated Output", expand=True))
    console.print()

    if csv_file:
        try:
            with open(csv_file, "w") as f:
                f.write("device,ttft_ms,throughput_tok_s,tokens_generated,total_time_sec\n")
                f.write(f"{device.upper()},{avg_ttft:.2f},{avg_tps:.2f},{int(avg_toks)},{avg_time:.3f}\n")
            console.print(f"[green]✔ Benchmark saved to {csv_file}[/green]\n")
        except Exception as e:
            console.print(f"[bold red]Failed to save CSV:[/bold red] {e}\n")


@app.command(name="bench", hidden=True)
def bench_alias(
    model_tag: str = typer.Argument(DEFAULT_MODEL),
    device: str = typer.Option("NPU", "--device", "-d"),
    tokens: int = typer.Option(128, "--tokens", "-t"),
    bench_iterations: int = typer.Option(1, "--bench-iterations"),
    csv_file: Optional[str] = typer.Option(None, "--csv"),
):
    """Alias for benchmark (FastFlowLM 'flm bench')."""
    benchmark(model_tag=model_tag, device=device, tokens=tokens, bench_iterations=bench_iterations, csv_file=csv_file)


@app.command(name="top")
def top(
    interval: float = typer.Option(1.0, "--interval", "-i", help="Refresh interval in seconds"),
    once: bool = typer.Option(False, "--once", "-1", help="Display single snapshot and exit"),
):
    """Real-time live hardware monitor for Intel NPU (frequency, power, memory, utilization)."""
    from intelflow.monitor import run_monitor
    run_monitor(interval=interval, once=once)


@app.command(name="monitor")
def monitor(
    interval: float = typer.Option(1.0, "--interval", "-i", help="Refresh interval in seconds"),
    once: bool = typer.Option(False, "--once", "-1", help="Display single snapshot and exit"),
):
    """Alias for 'inpu top' - real-time live hardware monitor."""
    from intelflow.monitor import run_monitor
    run_monitor(interval=interval, once=once)


@app.command()
def mcp():
    """Run the Model Context Protocol (MCP) server over stdio for Antigravity."""
    from intelflow.mcp_server import main as run_mcp
    run_mcp()


def main():
    app()


if __name__ == "__main__":
    main()
