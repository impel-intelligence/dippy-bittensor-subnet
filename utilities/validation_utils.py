import hashlib
import requests


def parse_size(line):
    """
    Parse the size string with unit and convert it to bytes.

    Args:
    - size_with_unit (str): The size string with unit (e.g., '125 MB')

    Returns:
    - int: The size in bytes
    """
    try:
        # get number enclosed in brackets
        size, unit = line[line.find("(") + 1 : line.rfind(")")].strip().split(" ")
        size = float(size.replace(",", ""))  # Remove commas for thousands
        unit = unit.lower()
        if unit == "kb":
            return int(size * 1024)
        elif unit == "mb":
            return int(size * 1024 * 1024)
        elif unit == "gb":
            return int(size * 1024 * 1024 * 1024)
        elif unit == "b":
            return int(size)  # No conversion needed for bytes
        else:
            raise ValueError(f"Unknown unit: {unit}")
    except ValueError as e:
        print(f"Error parsing size string '{size}{unit}': {e}")
        return 0


def regenerate_hash(namespace, name, chat_template , hotkey):
    s = " ".join([namespace, name, chat_template, hotkey])
    hash_output = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(hash_output[:16], 16)  # Returns a 64-bit integer from the first 16 hexadecimal characters
