import zlib

def file_checksum(filename):
    """Calculate CRC32 checksum of a file"""
    crc_val = 0
    with open(filename, 'rb') as f:
        # Read in chunks of 4K
        for chunk in iter(lambda: f.read(4096), b''):
            crc_val = zlib.crc32(chunk, crc_val)
    return format(crc_val & 0xFFFFFFFF, '08x') # Return as hex string, like sha256.hexdigest()
