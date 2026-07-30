"""Employee information service.

Provides employee data for leave request decisions. This is a mock
implementation using local data. In production, this would connect
to a real HR system.
"""

from __future__ import annotations

import structlog

from app.schemas.employee import EmployeeInfo, EmploymentStatus

logger = structlog.get_logger(__name__)


class EmployeeRepository:
    """Mock employee data repository.

    Contains fictional employee data for testing and demonstrations.
    Only includes fields relevant to leave decision logic.
    """

    # Mock employee database
    EMPLOYEES = {
        "EMP-001": {
            "employee_id": "EMP-001",
            "department": "Engineering",
            "employment_status": "ACTIVE",
            "leave_balance": 20,
            "manager_approval_required": True,
            "job_level": "senior",
        },
        "EMP-002": {
            "employee_id": "EMP-002",
            "department": "Finance",
            "employment_status": "ACTIVE",
            "leave_balance": 10,
            "manager_approval_required": True,
            "job_level": "standard",
        },
        "EMP-003": {
            "employee_id": "EMP-003",
            "department": "HR",
            "employment_status": "ACTIVE",
            "leave_balance": 25,
            "manager_approval_required": False,
            "job_level": "junior",
        },
        "EMP-004": {
            "employee_id": "EMP-004",
            "department": "Sales",
            "employment_status": "ON_LEAVE",
            "leave_balance": 5,
            "manager_approval_required": True,
            "job_level": "standard",
        },
    }

    def get_employee(self, employee_id: str) -> EmployeeInfo:
        """Retrieve employee information by ID.

        Args:
            employee_id: The employee's unique identifier.

        Returns:
            EmployeeInfo object with employee data.

        Raises:
            ValueError: If employee ID is invalid.
            KeyError: If employee not found.
        """
        if not employee_id or not employee_id.strip():
            logger.error("invalid_employee_id", employee_id=employee_id)
            raise ValueError("Employee ID cannot be empty")

        if employee_id not in self.EMPLOYEES:
            logger.warning("employee_not_found", employee_id=employee_id)
            raise KeyError(f"Employee not found: {employee_id}")

        data = self.EMPLOYEES[employee_id]
        employee = EmployeeInfo(
            employee_id=data["employee_id"],
            department=data["department"],
            employment_status=EmploymentStatus(data["employment_status"]),
            leave_balance=data["leave_balance"],
            manager_approval_required=data["manager_approval_required"],
            job_level=data["job_level"],
        )

        logger.info("employee_retrieved", employee_id=employee_id)
        return employee


class EmployeeService:
    """Service for accessing employee information.

    Provides a clean interface for LangGraph nodes to retrieve
    employee data without direct database access.
    """

    def __init__(self, repository: EmployeeRepository | None = None):
        """Initialize the employee service.

        Args:
            repository: EmployeeRepository instance. If None, uses default.
        """
        self.repository = repository or EmployeeRepository()
        logger.info("employee_service_initialized")

    def get_employee_info(self, employee_id: str) -> EmployeeInfo:
        """Retrieve employee information.

        Args:
            employee_id: The employee's unique identifier.

        Returns:
            EmployeeInfo object.

        Raises:
            ValueError: If employee ID is invalid.
            KeyError: If employee not found.
        """
        return self.repository.get_employee(employee_id)

    def validate_employee_id(self, employee_id: str) -> bool:
        """Check if an employee ID exists.

        Args:
            employee_id: The employee's unique identifier.

        Returns:
            True if employee exists, False otherwise.
        """
        try:
            self.repository.get_employee(employee_id)
            return True
        except (ValueError, KeyError):
            return False
