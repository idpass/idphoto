package org.idpass.idphoto

/**
 * Kotlin-idiomatic entry point for idphoto compression.
 *
 * Example usage:
 * ```kotlin
 * val result = IdPhoto.compress(photoBytes) {
 *     preset = Preset.QR_CODE
 *     maxDimension = 48
 * }
 * ```
 */
object IdPhoto {
    /**
     * Compress a photo with the given options.
     *
     * @param data Raw image bytes (JPEG, PNG, or WebP).
     * @param configure Optional DSL block to configure compression options.
     */
    fun compress(
        data: ByteArray,
        configure: CompressOptionsBuilder.() -> Unit = {},
    ): CompressedPhoto {
        val options = CompressOptionsBuilder().apply(configure).build()
        return compress(data, options)
    }

    /**
     * Compress a photo to fit within a byte budget.
     *
     * Quality is determined automatically via binary search and cannot
     * be specified manually; use [compress] if you need explicit quality control.
     *
     * @param data Raw image bytes (JPEG, PNG, or WebP).
     * @param maxBytes Maximum output size in bytes.
     * @param configure Optional DSL block to configure compression options.
     */
    fun compressToFit(
        data: ByteArray,
        maxBytes: Long,
        configure: CompressOptionsBuilder.() -> Unit = {},
    ): FitResult {
        require(maxBytes > 0) { "maxBytes must be positive" }
        val options = CompressOptionsBuilder().apply(configure).build()
        return compressToFit(data, maxBytes.toULong(), options)
    }
}

/**
 * Builder for [CompressOptions] used in the DSL-style API.
 *
 * All properties default to `null`, which means "use library default"
 * (or "use preset default" if a preset is set).
 */
class CompressOptionsBuilder {
    var preset: Preset? = null
    var maxDimension: Int? = null
    var quality: Float? = null
    var grayscale: Boolean? = null
    var cropMode: CropMode? = null
    var format: OutputFormat? = null
    var faceMargin: Float? = null

    internal fun build(): CompressOptions = CompressOptions(
        preset = preset,
        maxDimension = maxDimension?.toUInt(),
        quality = quality,
        grayscale = grayscale,
        cropMode = cropMode,
        format = format,
        faceMargin = faceMargin,
    )
}

// Convenience extension properties on CompressedPhoto

/** MIME type for the output format ("image/webp" or "image/jpeg"). */
val CompressedPhoto.mimeType: String
    get() = when (format) {
        OutputFormat.WEBP -> "image/webp"
        OutputFormat.JPEG -> "image/jpeg"
    }

/** File extension for the output format ("webp" or "jpg"). */
val CompressedPhoto.fileExtension: String
    get() = when (format) {
        OutputFormat.WEBP -> "webp"
        OutputFormat.JPEG -> "jpg"
    }

/** Human-readable summary that avoids dumping raw byte data to logs. */
fun CompressedPhoto.toSummaryString(): String =
    "CompressedPhoto(${width}x${height}, ${format.name.lowercase()}, ${data.size} bytes, original=${originalSize})"

/** Human-readable summary for fit results. */
fun FitResult.toSummaryString(): String =
    "FitResult(${photo.width}x${photo.height}, q=${qualityUsed}, reached=${reachedTarget}, ${photo.data.size} bytes)"
