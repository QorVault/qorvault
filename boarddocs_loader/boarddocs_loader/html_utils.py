"""BeautifulSoup helpers for extracting data from item innerHtml."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup


def extract_plain_text(html: str) -> str:
    """Strip all HTML tags, collapse whitespace, return clean text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_goals(html: str) -> list[str]:
    """Extract goal names from .goal elements with div.name children."""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    goals = []
    for goal_el in soup.find_all(class_="goal"):
        name_div = goal_el.find("div", class_="name")
        if name_div:
            text = name_div.get_text(strip=True)
            if text:
                goals.append(text)
    return goals


def extract_item_type(html: str) -> str:
    """Extract the Type value from dt/dd pairs in the innerHtml."""
    if not html:
        return "Information"
    soup = BeautifulSoup(html, "html.parser")
    for dt in soup.find_all("dt"):
        if dt.get_text(strip=True) == "Type":
            dd = dt.find_next_sibling("dd")
            if dd:
                val = dd.get_text(strip=True)
                return val if val else "Information"
    return "Information"
