"""API contract tests using schemathesis (property-based testing)."""
import pytest
import schemathesis


# Real OpenAPI schema test (requires server running)
@pytest.mark.skip("Requires server running")
@pytest.mark.integration
class TestOpenAPIContract:
    """Test API contract using property-based testing."""

    schema = schemathesis.from_uri("http://localhost:8000/openapi.json")

    @schema.parametrize()
    def test_api_properties(self, case):
        """Property-based API tests."""
        response = case.call()
        assert response.status_code in [200, 201, 400, 404, 500]