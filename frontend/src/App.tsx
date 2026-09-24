import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/layouts/AppShell";
import { CommandCenterPage } from "@/pages/CommandCenter";
import { CasesPage } from "@/pages/Cases";
import { CaseFilePage } from "@/pages/CaseFile";
import { PipelinePage } from "@/pages/Pipeline";
import { ThresholdStudioPage } from "@/pages/ThresholdStudio";
import { ModelPerformancePage } from "@/pages/ModelPerformance";
import { ReportsPage } from "@/pages/Reports";
import { SettingsPage } from "@/pages/Settings";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<CommandCenterPage />} />
        <Route path="/cases" element={<CasesPage />} />
        <Route path="/cases/:customerId" element={<CaseFilePage />} />
        <Route path="/pipeline" element={<PipelinePage />} />
        <Route path="/threshold" element={<ThresholdStudioPage />} />
        <Route path="/model" element={<ModelPerformancePage />} />
        <Route path="/reports" element={<ReportsPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
