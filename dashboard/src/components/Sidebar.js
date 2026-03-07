import React from "react";
import { Drawer, List, ListItem, ListItemText, Box, IconButton, useTheme, Divider } from "@mui/material";
import { Link, useLocation } from "react-router-dom";
import { Brightness4, Brightness7 } from "@mui/icons-material";
import { useTheme as useCustomTheme } from "../ThemeContext";

const Sidebar = () => {
  const location = useLocation();
  const muiTheme = useTheme();
  const { isDark, setIsDark } = useCustomTheme();

  const menuItems = [
    { label: "Dashboard", path: "/" },
    { label: "Hospitals", path: "/hospitals" },
    { label: "Model Metrics", path: "/model-metrics" },
    { label: "Federated Learning", path: "/federated-learning" },
    { label: "Blockchain Logs", path: "/blockchain-logs" },
    { label: "Security Alerts", path: "/security-alerts" }
  ];

  return (
    <Drawer 
      variant="permanent" 
      anchor="left"
      sx={{
        "& .MuiDrawer-paper": {
          backgroundColor: muiTheme.palette.background.paper,
          borderRight: `1px solid ${muiTheme.palette.divider}`,
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
        }
      }}
    >
      <Box sx={{ width: 260 }}>
        {/* Header */}
        <Box 
          sx={{ 
            p: 2.5, 
            background: `linear-gradient(135deg, ${muiTheme.palette.primary.main} 0%, ${muiTheme.palette.secondary.main} 100%)`, 
            color: "white",
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.2)"
          }}
        >
          <h3 style={{ margin: 0, fontSize: "1.4rem", fontWeight: 700, letterSpacing: "0.5px" }}>
            🏥 Healthcare AI
          </h3>
          <p style={{ margin: "4px 0 0 0", fontSize: "0.8rem", opacity: 0.9 }}>Federated Learning</p>
        </Box>

        {/* Theme Toggle */}
        <Box sx={{ p: 2, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontSize: "0.85rem", fontWeight: 500 }}>Theme</span>
          <IconButton
            size="small"
            onClick={() => setIsDark(!isDark)}
            sx={{
              color: muiTheme.palette.primary.main,
              backgroundColor: muiTheme.palette.mode === 'dark' ? '#2a2a3e' : '#f0f0f0',
              "&:hover": {
                backgroundColor: muiTheme.palette.mode === 'dark' ? '#3a3a4e' : '#e0e0e0',
              }
            }}
          >
            {isDark ? <Brightness7 fontSize="small" /> : <Brightness4 fontSize="small" />}
          </IconButton>
        </Box>

        <Divider sx={{ my: 1 }} />

        {/* Navigation Items */}
        <List sx={{ p: 0 }}>
          {menuItems.map((item) => (
            <ListItem
              button
              key={item.path}
              component={Link}
              to={item.path}
              sx={{
                background: location.pathname === item.path 
                  ? muiTheme.palette.mode === 'dark'
                    ? 'rgba(102, 126, 234, 0.1)'
                    : '#e8eaf6'
                  : "transparent",
                borderLeft: location.pathname === item.path 
                  ? `4px solid ${muiTheme.palette.primary.main}` 
                  : "4px solid transparent",
                color: location.pathname === item.path 
                  ? muiTheme.palette.primary.main 
                  : muiTheme.palette.text.primary,
                fontWeight: location.pathname === item.path ? 600 : 500,
                paddingLeft: "16px !important",
                paddingY: 1.5,
                marginY: 0.5,
                marginX: 1,
                borderRadius: location.pathname === item.path ? "8px" : "0",
                transition: "all 0.3s ease",
                "&:hover": {
                  background: muiTheme.palette.mode === 'dark'
                    ? 'rgba(102, 126, 234, 0.15)'
                    : 'rgba(102, 126, 234, 0.08)',
                  transform: "translateX(4px)",
                }
              }}
            >
              <ListItemText 
                primary={item.label}
                primaryTypographyProps={{
                  fontSize: "0.95rem",
                  fontWeight: "inherit"
                }}
              />
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  );
};

export default Sidebar;
