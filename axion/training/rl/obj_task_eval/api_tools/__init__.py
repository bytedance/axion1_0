__all__ = [
    "ArxivTool",
    "GoogleMapsTool",
    "PDFParser",
]


def __getattr__(name):
    if name == "ArxivTool":
        from .tool_arxiv import ArxivTool

        return ArxivTool
    if name == "GoogleMapsTool":
        from .tool_googlemap import GoogleMapsTool

        return GoogleMapsTool
    if name == "PDFParser":
        from .tool_pdf import PDFParser

        return PDFParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
