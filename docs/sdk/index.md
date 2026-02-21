# SDK Overview

Use this page to choose the right SDK.

## Choose Your SDK

| Language | Primary Entry Point | Best For |
| --- | --- | --- |
| Rust | `PhotoCompressor` builder | Native Rust services and pipelines |
| Python | `IdPhoto` + `CompressOptions`/`CompressToFitOptions` | Data science, backend scripting, ML workflows |
| TypeScript/JavaScript | `createIdPhoto()` + `IdPhoto` client | Browser and Node.js web apps |
| Kotlin | `IdPhoto.compress { ... }` DSL wrapper | Kotlin Android/JVM apps |
| Java | `JavaIdPhoto.compress(...)` static facade | Java Android/JVM apps |

## API Shape By Language

All SDKs expose the same core operations:

- `compress(...)`: one-shot compression with optional preset and overrides
- `compress_to_fit(...)` / `compressToFit(...)`: quality search to fit byte budgets

Each language uses conventions that match its ecosystem:

- Rust: fluent builder and enums
- Python: enums and typed option dataclasses
- JavaScript/TypeScript: client object and exported constants
- Kotlin: DSL-style configuration block
- Java: static facade with builder options and Java-native result models

## Quick Links

- [Rust SDK](rust.md)
- [Python SDK](python.md)
- [TypeScript/JavaScript SDK](wasm.md)
- [Kotlin SDK](kotlin.md)
- [Java SDK](java.md)

## Examples

=== "Rust"

    ```rust
    use idphoto::{PhotoCompressor, Preset};

    let input = std::fs::read("photo.jpg")?;
    let out = PhotoCompressor::new(input)?
        .preset(Preset::QrCode)
        .compress()?;
    # Ok::<(), idphoto::IdPhotoError>(())
    ```

=== "Python"

    ```python
    import idphoto

    raw = open("photo.jpg", "rb").read()
    result = idphoto.IdPhoto.compress(
        raw,
        options=idphoto.CompressOptions(preset=idphoto.Preset.QR_CODE),
    )
    ```

=== "JavaScript"

    ```javascript
    import { createIdPhoto, Preset } from "@idpass/idphoto-wasm";

    const idphoto = await createIdPhoto();
    const result = idphoto.compress(input, { preset: Preset.QR_CODE });
    ```

=== "Kotlin"

    ```kotlin
    val result = IdPhoto.compress(raw) {
        preset = Preset.QR_CODE
    }
    ```

=== "Java"

    ```java
    JavaCompressedPhoto result = JavaIdPhoto.compress(raw);
    ```
