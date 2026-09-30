"""Build and serve both Pages routes locally with the standard library."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from build_site import OUTPUT, build


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the DichromaticMap website locally")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    build()
    handler = partial(SimpleHTTPRequestHandler, directory=str(OUTPUT))
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Promotional page: http://127.0.0.1:{args.port}/", flush=True)
        print(f"Online app:       http://127.0.0.1:{args.port}/use.html", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
