"""Tests for the employee information service."""

from __future__ import annotations

import pytest
from app.services.employee import EmployeeService, EmployeeRepository
from app.schemas.employee import EmploymentStatus


class TestEmployeeService:
    """Tests for EmployeeService."""

    @pytest.fixture
    def employee_service(self):
        """Create an EmployeeService instance."""
        return EmployeeService()

    def test_get_employee_valid_id(self, employee_service):
        """Test retrieving an employee with a valid ID."""
        employee = employee_service.get_employee_info("EMP-001")

        assert employee.employee_id == "EMP-001"
        assert employee.department == "Engineering"
        assert employee.employment_status == EmploymentStatus.ACTIVE
        assert employee.leave_balance == 20
        assert employee.manager_approval_required is True

    def test_get_employee_unknown_id(self, employee_service):
        """Test that unknown employee ID raises KeyError."""
        with pytest.raises(KeyError):
            employee_service.get_employee_info("EMP-999")

    def test_get_employee_empty_id(self, employee_service):
        """Test that empty employee ID raises ValueError."""
        with pytest.raises(ValueError):
            employee_service.get_employee_info("")

    def test_get_employee_whitespace_id(self, employee_service):
        """Test that whitespace-only ID raises ValueError."""
        with pytest.raises(ValueError):
            employee_service.get_employee_info("   ")

    def test_validate_employee_id_valid(self, employee_service):
        """Test validating a known employee ID."""
        assert employee_service.validate_employee_id("EMP-001") is True

    def test_validate_employee_id_unknown(self, employee_service):
        """Test validating an unknown employee ID."""
        assert employee_service.validate_employee_id("EMP-999") is False

    def test_validate_employee_id_empty(self, employee_service):
        """Test validating an empty employee ID."""
        assert employee_service.validate_employee_id("") is False

    def test_multiple_employees_in_repository(self, employee_service):
        """Test that multiple employees are available."""
        emp1 = employee_service.get_employee_info("EMP-001")
        emp2 = employee_service.get_employee_info("EMP-002")
        emp3 = employee_service.get_employee_info("EMP-003")

        assert emp1.employee_id != emp2.employee_id
        assert emp2.employee_id != emp3.employee_id

    def test_employee_on_leave_status(self, employee_service):
        """Test retrieving an employee with ON_LEAVE status."""
        employee = employee_service.get_employee_info("EMP-004")

        assert employee.employment_status == EmploymentStatus.ON_LEAVE

    def test_employee_model_validation(self, employee_service):
        """Test that employee data is properly validated."""
        employee = employee_service.get_employee_info("EMP-002")

        # Verify all required fields are present
        assert hasattr(employee, "employee_id")
        assert hasattr(employee, "department")
        assert hasattr(employee, "employment_status")
        assert hasattr(employee, "leave_balance")
        assert hasattr(employee, "manager_approval_required")
