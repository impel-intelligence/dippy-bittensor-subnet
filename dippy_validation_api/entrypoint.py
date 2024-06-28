
import typer
app = typer.Typer(rich_markup_mode="rich")

def _run_eval():

    pass

def _run_vibe():
    pass


@app.command()
def run(
    path: Annotated[
        Union[str, None],
        typer.Argument(
            help="A path to a Python file or package directory (with [blue]__init__.py[/blue] files) containing a [bold]FastAPI[/bold] app. If not provided, a default set of paths will be tried."
        ),
    ] = None,):

    _run_eval()
    return
