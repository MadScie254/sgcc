import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/layouts/AppShell";
import { OverviewPage } from "@/pages/Overview";
import { AnalyticsPage } from "@/pages/Analytics";
import { PredictPage } from "@/pages/Predict";
import { ExplainPage } from "@/pages/Explain";
import { MonitorPage } from "@/pages/Monitor";
import { CustomersPage } from "@/pages/Customers";
import { SettingsPage } from "@/pages/Settings";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/explain" element={<ExplainPage />} />
        <Route path="/monitor" element={<MonitorPage />} />
        <Route path="/customers" element={<CustomersPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}