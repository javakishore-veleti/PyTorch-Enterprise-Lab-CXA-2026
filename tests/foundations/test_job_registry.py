"""Unit tests for JobRegistry."""
from quantedge_services.core.jobs import JobRegistry, JobStatus


class TestJobRegistry:
    def setup_method(self):
        self._registry = JobRegistry()

    def test_create_returns_pending_job(self):
        job = self._registry.create("test_task")
        assert job.status == JobStatus.PENDING
        assert job.task_name == "test_task"
        assert job.id is not None

    def test_get_returns_job_by_id(self):
        job = self._registry.create("test_task")
        fetched = self._registry.get(job.id)
        assert fetched is not None
        assert fetched.id == job.id

    def test_set_running_updates_status(self):
        job = self._registry.create("test_task")
        self._registry.set_running(job.id)
        assert self._registry.get(job.id).status == JobStatus.RUNNING
        assert self._registry.get(job.id).started_at is not None

    def test_set_success_updates_result(self):
        job = self._registry.create("test_task")
        self._registry.set_running(job.id)
        self._registry.set_success(job.id, {"key": "value"})
        record = self._registry.get(job.id)
        assert record.status == JobStatus.SUCCESS
        assert record.result == {"key": "value"}
        assert record.completed_at is not None

    def test_set_failed_stores_error(self):
        job = self._registry.create("test_task")
        self._registry.set_running(job.id)
        self._registry.set_failed(job.id, "something went wrong")
        record = self._registry.get(job.id)
        assert record.status == JobStatus.FAILED
        assert record.error == "something went wrong"

    def test_get_unknown_job_returns_none(self):
        assert self._registry.get("nonexistent-id") is None

    def test_list_by_task_filters_correctly(self):
        self._registry.create("task_a")
        self._registry.create("task_a")
        self._registry.create("task_b")
        results = self._registry.list_all(task_name="task_a")
        assert len(results) == 2

    def test_list_all_with_status_filter(self) -> None:
        j1 = self._registry.create("task_a")
        j2 = self._registry.create("task_a")
        self._registry.set_running(j1.id)
        self._registry.set_success(j1.id, {})
        results = self._registry.list_all(status=JobStatus.SUCCESS)
        assert len(results) == 1
        assert results[0].id == j1.id

    def test_list_all_newest_first(self) -> None:
        j1 = self._registry.create("task_a")
        j2 = self._registry.create("task_a")
        results = self._registry.list_all()
        # j2 was created after j1 — should appear first
        assert results[0].id == j2.id

    def test_list_all_no_filter_returns_all(self) -> None:
        self._registry.create("task_x")
        self._registry.create("task_y")
        assert len(self._registry.list_all()) == 2
