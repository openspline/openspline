"""Show the next steps after the native installer finishes."""

import shlex
import sys
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text


def print_install_message(install_dir: Path, console: Console | None = None) -> None:
    console = console or Console()
    run_command = f"sh {shlex.quote(str(install_dir / 'run.sh'))}"
    message = Group(
        Text.assemble("Installed in ", (str(install_dir), "bold")),
        Text(""),
        Text("Launch the demo", style="bold"),
        Text(run_command, style="bold cyan"),
        Text("Open http://localhost:7860 after it starts (or your configured port)."),
        Text(""),
        Text("Realtime conversation (optional)", style="bold"),
        Text.assemble("Set one or both keys in ", (str(install_dir / ".env"), "cyan"), ":"),
        Text("OPENAI_API_KEY=your_openai_key", style="cyan"),
        Text("GEMINI_API_KEY=your_gemini_key", style="cyan"),
        Text("Restart the demo after editing .env. Audio file mode needs no key."),
    )
    console.print(Panel(message, title="openspline installation complete", border_style="green"))


def main() -> None:
    print_install_message(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
