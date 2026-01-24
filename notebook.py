# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from src.config import Config
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
