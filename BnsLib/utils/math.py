def safe_min(inp):
    try:
        return min(inp)
    except TypeError:
        return min([inp])

def safe_max(inp):
    try:
        return max(inp)
    except TypeError:
        return max([inp])
