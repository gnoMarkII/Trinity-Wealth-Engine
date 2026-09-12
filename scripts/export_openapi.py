"""Export FastAPI OpenAPI schema to static JSON for TypeScript type generation."""
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


def main():
    schema = app.openapi()
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web", "openapi.json"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Exported OpenAPI schema to {output_path} successfully ({len(schema.get('components', {}).get('schemas', {}))} schemas)")


if __name__ == "__main__":
    main()
