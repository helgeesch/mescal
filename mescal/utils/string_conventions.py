from typing import Iterable, Callable
import re
from collections import Counter
from enum import Enum


class StringConventionEnum(Enum):
    PascalCase = 'PascalCase'
    camelCase = 'camelCase'
    Title__Space = 'Title Space'
    SCREAMING_SNAKE_CASE = 'SCREAMING_SNAKE_CASE'
    lower_snake = 'lower_snake'
    NoConvention = 'No convention'


_CONVENTION_PATTERNS = {
    StringConventionEnum.PascalCase: r"^[A-Z][a-zA-Z0-9]*$",
    StringConventionEnum.camelCase: r"^[a-z][a-zA-Z0-9]*$",
    StringConventionEnum.Title__Space: r"^([A-Z][a-z]+)(\s[A-Z][a-z]+)*$",
    StringConventionEnum.SCREAMING_SNAKE_CASE: r"^[A-Z0-9]+(_[A-Z0-9]+)*$",
    StringConventionEnum.lower_snake: r"^[a-z0-9]+(_[a-z0-9]+)*$",
}


def identify_string_convention(strings: str | Iterable[str]) -> StringConventionEnum:
    if isinstance(strings, str):
        return _identify_convention_for_single_string(strings)
    return _identify_convention_for_set_of_strings(strings)


def _identify_convention_for_set_of_strings(strings: Iterable[str]) -> StringConventionEnum:
    if not strings:
        return StringConventionEnum.NoConvention

    matches = Counter()
    for string in strings:
        for convention, pattern in _CONVENTION_PATTERNS.items():
            if re.match(pattern, string):
                matches[convention] += 1
                break

    if matches:
        most_common = matches.most_common(1)[0]
        return most_common[0] if most_common[1] > 0 else None
    return StringConventionEnum.NoConvention


def _identify_convention_for_single_string(string: str) -> StringConventionEnum:
    for convention, pattern in _CONVENTION_PATTERNS.items():
        if re.match(pattern, string):
            return convention

    return StringConventionEnum.NoConvention


def get_translation_method_to(convention: StringConventionEnum) -> Callable[[str], str]:
    if convention == StringConventionEnum.PascalCase:
        return to_pascal_case
    elif convention == StringConventionEnum.camelCase:
        return to_camel_case
    elif convention == StringConventionEnum.Title__Space:
        return to_title_space
    elif convention == StringConventionEnum.SCREAMING_SNAKE_CASE:
        return to_screaming_snake_case
    elif convention == StringConventionEnum.lower_snake:
        return to_lower_snake
    else:
        return lambda x: x


def add_prefix_to_string_in_same_convention(string: str, prefix: str) -> str:
    convention = _identify_convention_for_single_string(string)

    if convention == StringConventionEnum.PascalCase:
        return f"{prefix.capitalize()}{string}"
    elif convention == StringConventionEnum.camelCase:
        return f"{prefix.lower()}{string[0].upper()}{string[1:]}"
    elif convention == StringConventionEnum.Title__Space:
        return f"{prefix.capitalize()} {string}"
    elif convention == StringConventionEnum.SCREAMING_SNAKE_CASE:
        return f"{prefix.upper()}_{string}"
    elif convention == StringConventionEnum.lower_snake:
        return f"{prefix.lower()}_{string}"
    else:
        return string  # No convention identified, return as is


def add_suffix_to_string_in_same_convention(string: str, suffix: str) -> str:
    convention = _identify_convention_for_single_string(string)

    if convention == StringConventionEnum.PascalCase:
        return f"{string}{suffix.capitalize()}"
    elif convention == StringConventionEnum.camelCase:
        return f"{string}{suffix[0].upper()}{suffix[1:]}"
    elif convention == StringConventionEnum.Title__Space:
        return f"{string} {suffix.capitalize()}"
    elif convention == StringConventionEnum.SCREAMING_SNAKE_CASE:
        return f"{string}_{suffix.upper()}"
    elif convention == StringConventionEnum.lower_snake:
        return f"{string}_{suffix.lower()}"
    else:
        return string  # No convention identified, return as is


def to_lower_snake(text: str) -> str:
    """Convert PascalCase, camelCase, Title Space, SCREAMING_SNAKE_CASE to lower_snake."""
    text = re.sub(r'(?<=[a-z])([A-Z])', r'_\1', text)
    text = re.sub(r'(?<=[A-Z])([A-Z][a-z])', r'_\1', text)
    text = re.sub(r'[\s]+', '_', text)
    return text.lower()


def to_title_space(text: str) -> str:
    """Convert PascalCase, camelCase, lower_snake, SCREAMING_SNAKE_CASE to Title Space."""
    text = re.sub(r'(_|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z]))', ' ', text)
    words = text.split()
    return ' '.join([word if word.isupper() else word.capitalize() for word in words])


def to_pascal_case(text: str) -> str:
    """Convert camelCase, Title Space, lower_snake, SCREAMING_SNAKE_CASE to PascalCase."""
    text = re.sub(r'(_|(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z]))', ' ', text)
    words = text.split()
    return ''.join([word if word.isupper() else word.capitalize() for word in words])


def to_camel_case(text: str) -> str:
    """Convert PascalCase, Title Space, lower_snake, SCREAMING_SNAKE_CASE to camelCase."""
    pascal_case = to_pascal_case(text)
    return pascal_case[0].lower() + pascal_case[1:] if pascal_case else ''


def to_screaming_snake_case(text: str) -> str:
    """Convert PascalCase, camelCase, Title Space, lower_snake, to SCREAMING_SNAKE_CASE."""
    lower_snake = to_lower_snake(text)
    return lower_snake.upper()


if __name__ == '__main__':
    print(to_lower_snake('MyTextABCBing'))     # Output: my_text_abc_bing
    print(to_lower_snake('myTextABCBing'))     # Output: my_text_abc_bing
    print(to_lower_snake('My Text ABC Bing'))  # Output: my_text_abc_bing
    print(to_lower_snake('my_text_ABC_Bing'))  # Output: my_text_abc_bing
    print(to_lower_snake('MY_TEXT_ABC_BING'))  # Output: my_text_abc_bing

    print(to_title_space('MyTextABCBing'))     # Output: My Text ABC Bing
    print(to_title_space('myTextABCBing'))     # Output: My Text ABC Bing
    print(to_title_space('My Text ABC Bing'))  # Output: My Text ABC Bing
    print(to_title_space('my_text_ABC_Bing'))  # Output: My Text ABC Bing
    print(to_title_space('MY_TEXT_ABC_BING'))  # Output: My Text ABC Bing

    print(to_pascal_case('MyTextABCBing'))     # Output: MyTextABCBing
    print(to_pascal_case('myTextABCBing'))     # Output: MyTextABCBing
    print(to_pascal_case('My Text ABC Bing'))  # Output: MyTextABCBing
    print(to_pascal_case('my_text_ABC_Bing'))  # Output: MyTextABCBing
    print(to_pascal_case('MY_TEXT_ABC_BING'))  # Output: MyTextABCBing

    print(to_camel_case('MyTextABCBing'))     # Output: myTextABCBing
    print(to_camel_case('myTextABCBing'))     # Output: myTextABCBing
    print(to_camel_case('My Text ABC Bing'))  # Output: myTextABCBing
    print(to_camel_case('my_text_ABC_Bing'))  # Output: myTextABCBing
    print(to_camel_case('MY_TEXT_ABC_BING'))  # Output: myTextABCBing

    print(to_screaming_snake_case('MyTextABCBing'))     # Output: MY_TEXT_ABC_BING
    print(to_screaming_snake_case('myTextABCBing'))     # Output: MY_TEXT_ABC_BING
    print(to_screaming_snake_case('My Text ABC Bing'))  # Output: MY_TEXT_ABC_BING
    print(to_screaming_snake_case('my_text_ABC_Bing'))  # Output: MY_TEXT_ABC_BING
    print(to_screaming_snake_case('MY_TEXT_ABC_BING'))  # Output: MY_TEXT_ABC_BING

    sample_strings = [
        "PascalCaseExample", "camelCaseExample", "AnotherPascalCase",
        "Title Space Example", "SCREAMING_SNAKE_CASE", "lower_snake_case"
    ]

    dominant_convention = identify_string_convention(sample_strings)
    print(f"Dominant convention in list: {dominant_convention}")

    for single_string in sample_strings:
        string_convention = identify_string_convention(single_string)
        print(f"Convention for single string '{single_string}': {string_convention}")