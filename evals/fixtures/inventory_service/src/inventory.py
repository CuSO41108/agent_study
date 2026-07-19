def reserve(stock: int, quantity: int) -> int:
    if quantity < 0:
        raise ValueError("quantity must be non-negative")
    if quantity > stock:
        raise ValueError("insufficient stock")
    return stock - quantity


def is_low_stock(stock: int, threshold: int) -> bool:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return stock < threshold
