# Chat With PDF 0.0.1

A smart PDF analysis application that extracts and classifies document content using advanced text processing and local LLM integration.

## Features

### Content Type Classification

The application automatically analyzes and classifies text blocks in PDF documents using frequency-based font analysis:

#### Classification Rules

1. **Document Analysis**: The system first analyzes font size frequency across the entire document
2. **Body Text Identification**: The most frequently used font size is classified as `body_text`
3. **Content Type Assignment**:
   - **Heading**: Largest font size in the document (above body text size)
   - **Sub-heading**: Any font size between heading and body text
   - **Body Text**: Most frequent font size (baseline for classification)
   - **Footnote**: Smallest font size in the document (below body text size)
   - **Other**: Any remaining font sizes below body text (but not the smallest)
   - **List Item**: Text blocks starting with bullets (`•`, `-`, `*`), numbers (`1.`, `2)`), or letters (`a.`, `b)`)
   - **Mixed Content**: Blocks containing multiple font sizes or styles

#### Font Style Detection

The system also analyzes font styling flags from PyMuPDF:
- **Plain Text** (flag 4): Regular text
- **Italic Text** (flag 6): Emphasized text
- **Bold Text** (flag 20): Strong emphasis

#### Supported List Patterns

- Bullet points: `•`, `-`, `*`, `◦`
- Numbered lists: `1`, `1.`, `1)`, `(1)`
- Lettered lists: `a`, `a.`, `a)`, `(a)`
- Supports up to 99 numbered items and a-z lettered items

## Run the app

### uv

Run as a desktop app:

```
uv run flet run
```

Run as a web app:

```
uv run flet run --web
```

### Poetry

Install dependencies from `pyproject.toml`:

```
poetry install
```

Run as a desktop app:

```
poetry run flet run
```

Run as a web app:

```
poetry run flet run --web
```

For more details on running the app, refer to the [Getting Started Guide](https://flet.dev/docs/getting-started/).

## Build the app

### Android

```
flet build apk -v
```

For more details on building and signing `.apk` or `.aab`, refer to the [Android Packaging Guide](https://flet.dev/docs/publish/android/).

### iOS

```
flet build ipa -v
```

For more details on building and signing `.ipa`, refer to the [iOS Packaging Guide](https://flet.dev/docs/publish/ios/).

### macOS

```
flet build macos -v
```

For more details on building macOS package, refer to the [macOS Packaging Guide](https://flet.dev/docs/publish/macos/).

### Linux

```
flet build linux -v
```

For more details on building Linux package, refer to the [Linux Packaging Guide](https://flet.dev/docs/publish/linux/).

### Windows

```
flet build windows -v
```

For more details on building Windows package, refer to the [Windows Packaging Guide](https://flet.dev/docs/publish/windows/).