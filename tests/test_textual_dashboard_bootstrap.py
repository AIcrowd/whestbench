from circuit_estimation.textual_dashboard.app import DashboardApp


def test_dashboard_app_importable() -> None:
    app = DashboardApp(report={})
    assert app is not None
