import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Box } from "@mui/material";
import { ThemeProvider } from "./ThemeContext";
import Sidebar from "./components/Sidebar";
import Dash from "./pages/Dash";
import Hospitals from "./pages/Hospitals";
import ModelMetrics from "./pages/ModelMetrics";
import BlockchainLogs from "./pages/BlockchainLogs";
import SecurityAlerts from "./pages/SecurityAlerts";
import FederatedLearning from "./pages/FederatedLearning";

function AppContent() {
  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      <Sidebar />
      <Box sx={{ flex: 1, overflow: "auto" }}>
        <Routes>
          <Route path="/" element={<Dash />} />
          <Route path="/hospitals" element={<Hospitals />} />
          <Route path="/model-metrics" element={<ModelMetrics />} />
          <Route path="/blockchain-logs" element={<BlockchainLogs />} />
          <Route path="/security-alerts" element={<SecurityAlerts />} />
          <Route path="/federated-learning" element={<FederatedLearning />} />
        </Routes>
      </Box>
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppContent />
      </Router>
    </ThemeProvider>
  );
}

export default App;
