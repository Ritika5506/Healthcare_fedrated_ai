import React from "react";
import { Drawer, List, ListItem, ListItemText } from "@mui/material";

const Sidebar = () => {
  return (
    <Drawer variant="permanent" anchor="left">
      <List sx={{ width: 220 }}>
        <ListItem button>
          <ListItemText primary="Dashboard" />
        </ListItem>

        <ListItem button>
          <ListItemText primary="Hospitals" />
        </ListItem>

        <ListItem button>
          <ListItemText primary="Model Metrics" />
        </ListItem>

        <ListItem button>
          <ListItemText primary="Blockchain Logs" />
        </ListItem>

        <ListItem button>
          <ListItemText primary="Security Alerts" />
        </ListItem>
      </List>
    </Drawer>
  );
};

export default Sidebar;
