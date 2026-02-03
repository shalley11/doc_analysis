def sort_blocks_reading_order(blocks):
    """
    Sort blocks in reading order.
    Falls back safely when bbox is missing.
    """

    def sort_key(b):
        bbox = b.get("bbox")

        if bbox and len(bbox) == 4:
            # y first, then x
            return (b["page_no"], round(bbox[1], 1), round(bbox[0], 1))

        # fallback: page order only
        return (b["page_no"], float("inf"), float("inf"))

    return sorted(blocks, key=sort_key)
