/// Strip the ICC color profile (ICCP chunk) from a WebP file.
///
/// WebP encoders often embed an sRGB ICC profile (~456 bytes) via the VP8X
/// extended format. At small dimensions (e.g. 48x64) this metadata can exceed
/// the actual image data. This function extracts the VP8/VP8L bitstream and
/// rewraps it in a minimal RIFF container, discarding all extended chunks.
///
/// Returns the input unchanged if:
/// - The data is too small to be a valid WebP
/// - The RIFF/WEBP signature is missing
/// - The file is already in simple format (VP8 or VP8L, no VP8X)
/// - No VP8/VP8L chunk is found within the extended format
pub fn strip_icc_profile(data: &[u8]) -> Vec<u8> {
    // Minimum valid WebP: 12-byte RIFF header + at least one 8-byte chunk header
    if data.len() < 20 {
        return data.to_vec();
    }

    // Verify RIFF signature
    if &data[0..4] != b"RIFF" {
        return data.to_vec();
    }

    // Verify WEBP signature
    if &data[8..12] != b"WEBP" {
        return data.to_vec();
    }

    // Check first chunk FourCC at offset 12
    let first_chunk = &data[12..16];
    if first_chunk != b"VP8X" {
        // Already simple format (VP8 or VP8L) — no stripping needed
        return data.to_vec();
    }

    // Walk chunks after VP8X to find the VP8/VP8L image data chunk.
    // VP8X chunk: 4 (fourCC) + 4 (size) + 10 (payload) = 18 bytes
    let mut offset: usize = 12 + 4 + 4 + 10; // skip RIFF header (12) + VP8X chunk (18)

    while offset + 8 <= data.len() {
        let chunk_fourcc = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;
        // RIFF chunks are padded to even size
        let padded_size = chunk_size + (chunk_size % 2);

        if chunk_fourcc == b"VP8 " || chunk_fourcc == b"VP8L" {
            // Found the image chunk — rewrap in a minimal RIFF container
            let chunk_total_size = 8 + padded_size; // fourCC + size + data
            let riff_payload_size = 4 + chunk_total_size; // "WEBP" + chunk

            let mut result = Vec::with_capacity(12 + chunk_total_size);

            // RIFF header
            result.extend_from_slice(b"RIFF");
            result.extend_from_slice(&(riff_payload_size as u32).to_le_bytes());
            result.extend_from_slice(b"WEBP");

            // Copy VP8/VP8L chunk as-is
            let chunk_end = (offset + chunk_total_size).min(data.len());
            result.extend_from_slice(&data[offset..chunk_end]);

            return result;
        }

        offset += 8 + padded_size;
    }

    // No VP8/VP8L chunk found — return original
    data.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_small_returns_unchanged() {
        let data = vec![0u8; 10];
        assert_eq!(strip_icc_profile(&data), data);
    }

    #[test]
    fn non_riff_returns_unchanged() {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(b"NOPE");
        assert_eq!(strip_icc_profile(&data), data);
    }

    #[test]
    fn non_webp_returns_unchanged() {
        let mut data = vec![0u8; 24];
        data[0..4].copy_from_slice(b"RIFF");
        data[8..12].copy_from_slice(b"NOPE");
        assert_eq!(strip_icc_profile(&data), data);
    }

    #[test]
    fn simple_vp8_returns_unchanged() {
        // Build a minimal simple-format WebP (VP8 directly, no VP8X)
        let image_data = vec![0xAA; 10];
        let chunk_size = image_data.len() as u32;
        let riff_size = 4 + 4 + 4 + chunk_size; // WEBP + VP8_ fourcc + size + data

        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&riff_size.to_le_bytes());
        data.extend_from_slice(b"WEBP");
        data.extend_from_slice(b"VP8 ");
        data.extend_from_slice(&chunk_size.to_le_bytes());
        data.extend_from_slice(&image_data);

        assert_eq!(strip_icc_profile(&data), data);
    }

    #[test]
    fn vp8x_with_iccp_strips_to_vp8() {
        // Build a VP8X WebP with an ICCP chunk followed by VP8
        let image_data = vec![0xBB; 16];
        let icc_data = vec![0xCC; 456]; // typical ICC profile

        let mut data = Vec::new();
        // RIFF header (placeholder size)
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&[0; 4]); // will fix later
        data.extend_from_slice(b"WEBP");

        // VP8X chunk (10-byte payload)
        data.extend_from_slice(b"VP8X");
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 10]); // VP8X payload (flags + dimensions)

        // ICCP chunk
        data.extend_from_slice(b"ICCP");
        data.extend_from_slice(&(icc_data.len() as u32).to_le_bytes());
        data.extend_from_slice(&icc_data);

        // VP8 chunk
        data.extend_from_slice(b"VP8 ");
        data.extend_from_slice(&(image_data.len() as u32).to_le_bytes());
        data.extend_from_slice(&image_data);

        // Fix RIFF size
        let riff_size = (data.len() - 8) as u32;
        data[4..8].copy_from_slice(&riff_size.to_le_bytes());

        let result = strip_icc_profile(&data);

        // Result should be a minimal RIFF with just the VP8 chunk
        assert!(result.len() < data.len(), "stripped should be smaller");
        assert_eq!(&result[0..4], b"RIFF");
        assert_eq!(&result[8..12], b"WEBP");
        assert_eq!(&result[12..16], b"VP8 ");
        // The VP8 image data should be preserved
        let result_image_offset = 20; // 12 (RIFF header) + 4 (VP8 ) + 4 (size)
        assert_eq!(
            &result[result_image_offset..result_image_offset + image_data.len()],
            &image_data
        );
    }

    #[test]
    fn vp8x_with_iccp_strips_to_vp8l() {
        // VP8X with ICCP followed by VP8L (lossless)
        let image_data = vec![0xDD; 20];

        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(b"WEBP");

        // VP8X chunk
        data.extend_from_slice(b"VP8X");
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 10]);

        // ICCP chunk (odd size to test padding)
        let icc_data = vec![0xEE; 457]; // odd size
        data.extend_from_slice(b"ICCP");
        data.extend_from_slice(&(icc_data.len() as u32).to_le_bytes());
        data.extend_from_slice(&icc_data);
        data.push(0); // padding byte for even alignment

        // VP8L chunk
        data.extend_from_slice(b"VP8L");
        data.extend_from_slice(&(image_data.len() as u32).to_le_bytes());
        data.extend_from_slice(&image_data);

        // Fix RIFF size
        let riff_size = (data.len() - 8) as u32;
        data[4..8].copy_from_slice(&riff_size.to_le_bytes());

        let result = strip_icc_profile(&data);

        assert!(result.len() < data.len());
        assert_eq!(&result[12..16], b"VP8L");
    }

    #[test]
    fn vp8x_without_image_chunk_returns_unchanged() {
        // VP8X with only an ICCP chunk, no VP8/VP8L
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&[0; 4]);
        data.extend_from_slice(b"WEBP");

        data.extend_from_slice(b"VP8X");
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 10]);

        data.extend_from_slice(b"ICCP");
        let icc = vec![0xFF; 100];
        data.extend_from_slice(&(icc.len() as u32).to_le_bytes());
        data.extend_from_slice(&icc);

        let riff_size = (data.len() - 8) as u32;
        data[4..8].copy_from_slice(&riff_size.to_le_bytes());

        let result = strip_icc_profile(&data);
        assert_eq!(result, data);
    }
}
