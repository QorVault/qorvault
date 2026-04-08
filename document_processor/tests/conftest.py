"""Shared test fixtures for document_processor tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_html() -> str:
    return """
    <div class="content">
        <h2>Board Resolution 2024-15</h2>
        <p>The Kent School District Board of Directors hereby approves
        the annual budget for fiscal year 2024-2025. This resolution
        authorizes the superintendent to allocate funds as outlined
        in the attached budget document.</p>
        <p>The total operating budget shall not exceed $450 million.
        Capital expenditures are approved at $85 million for facility
        improvements across the district.</p>
    </div>
    """


@pytest.fixture
def long_text() -> str:
    """Generate text long enough to produce multiple chunks."""
    paragraph = (
        "The Kent School District Board of Directors met in regular session "
        "to discuss the annual budget proposal for the upcoming fiscal year. "
        "Board members reviewed the superintendent's recommendations for "
        "staffing levels, facility maintenance, and curriculum development. "
        "The finance committee presented a detailed analysis of revenue "
        "projections based on state funding formulas and local property "
        "tax assessments. Community members provided public comment on "
        "priorities including class size reduction and technology upgrades. "
    )
    # Repeat enough times to exceed 384 tokens multiple times
    return " ".join([paragraph] * 20)
