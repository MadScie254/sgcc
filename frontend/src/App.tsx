import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { DashboardPage } from "@/routes/DashboardPage";
import { EdaPage } from "@/routes/EdaPage";
import { PredictPage } from "@/routes/PredictPage";
import { ExplainPage } from "@/routes/ExplainPage";
import { UploadPage } from "@/routes/UploadPage";
import { ComparePage } from "@/routes/ComparePage";
import { MonitorPage } from "@/routes/MonitorPage";
import { TrainPage } from "@/routes/TrainPage";
import { ResearchValidationPage } from "@/routes/ResearchValidationPage";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/eda" element={<EdaPage />} />
        <Route path="/train" element={<TrainPage />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/explain" element={<ExplainPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/monitor" element={<MonitorPage />} />
        <Route path="/research-validation" element={<ResearchValidationPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}