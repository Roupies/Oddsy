Of course. Here is a revised and context-rich Project Charter for Oddsy, adapted with the full history and specific technical decisions of your project.

---
# **PROJECT CHARTER**
## **Oddsy - EPL Prediction & Analytics Platform**

**Project Lead:** Maxime Naguet  
**Date:** September 26, 2025  
**Document Version:** 2.0 (Live Project Phase)

---
## **PROJECT OBJECTIVES**

### **Purpose**
To engineer and deploy a sophisticated, production-grade machine learning system that delivers reliable English Premier League (EPL) match predictions. This project addresses the challenge of creating a sports analytics tool that not only surpasses naive baselines but also provides actionable insights through a robust, full-stack application, moving beyond academic accuracy to real-world utility and validation.

### **SMART Objectives**

1.  **Develop a "Champion" Cascade Model:** Evolve the existing model into a specialized two-forest cascade architecture, aiming to increase the **draw recall to over 30%** while maintaining a **global accuracy of >52%** on the live 2025-26 EPL season.
2.  **Build a Scalable Full-Stack Infrastructure:** Implement a **PostgreSQL** database, a **FastAPI** backend, and a **React** frontend, deployed on a hybrid Vercel/Render architecture, capable of serving live predictions with an API response time of under 500ms by Week 10.
3.  **Automate the Live Data Pipeline:** Create a fully automated weekly pipeline that programmatically downloads the latest match data, recalculates all features with guaranteed temporal integrity, and updates the database without manual intervention.

---
## **STAKEHOLDERS AND TEAM ROLES**

### **Internal Stakeholders**
* **Maxime Naguet** - Solo Project Lead
    * **Project Manager:** Defines the roadmap, tracks progress against milestones, manages risks.
    * **ML Engineer:** Designs, trains, and validates the specialized cascade models; develops the feature engineering pipeline.
    * **Backend Developer:** Implements the PostgreSQL database schema and the FastAPI endpoints.
    * **Frontend Developer:** Develops the interactive React dashboard.
    * **DevOps Engineer:** Manages the Dockerized database, CI/CD, and deployment on Render/Vercel.

### **External Stakeholders**
* **Course Instructors:** Provide academic evaluation, feedback, and project guidance.
* **End Users:** (Target Audience) Sports analysts, betting enthusiasts, and data science students interacting with the final dashboard.
* **Data Providers:** Football-Data.co.uk (for historical CSVs and ongoing results).

---
## **PROJECT SCOPE**

### **In-Scope**
* **ML Model:** A specialized two-forest cascade model for H/D/A prediction, including a dedicated "Draw Specialist" forest.
* **Database:** A Dockerized PostgreSQL database to serve as the single source of truth for all historical and live data.
* **Backend:** A FastAPI application to serve model predictions and analytical data.
* **Frontend:** An interactive dashboard built with React for data visualization and prediction display.
* **Automation:** A weekly cron job to automatically download new data, recalculate features, and update predictions.
* **Validation:** Continuous performance tracking on the live EPL 2025-26 season.

### **Out-of-Scope**
* Predictions for leagues other than the EPL.
* Direct betting or financial transaction capabilities.
* User authentication or personalized accounts in the initial version.
* Advanced features like player-level data or real-time odds movement analysis (reserved for future R&D).

---
## **RISK ASSESSMENT AND MITIGATION**

| Risk Category | Risk Description | Probability | Impact | Mitigation Strategy |
|---------------|------------------|-------------|---------|-------------------|
| **Technical** | Final model accuracy remains below the 50% "Good" target on live data. | Medium | High | Focus on the specialized two-forest architecture; use a simplified binary model (Home vs. Not-Home) as a fallback if necessary. |
| **Data** | The external data source (Football-Data.co.uk) changes its URL or format, breaking the automated pipeline. | Medium | Medium | Implement robust error handling and logging in the download script; have a manual download process as a backup. |
| **Timeline** | Integrating the database, backend, and frontend takes longer than the allocated time. | High | Medium | Follow a strict "API-first" design contract; use placeholder (mock) API data for frontend development to work in parallel. |
| **Resource** | As a solo developer, balancing four distinct roles (ML, Backend, Frontend, DevOps) is challenging. | High | Medium | Prioritize ruthlessly: a functional backend is more critical than a polished UI. Leverage pre-built component libraries (e.g., Material-UI for React). |
| **Deployment** | Free tier limitations on Render (e.g., DB expiration, service spin-down) impact the user experience. | Medium | Low | Clearly document these limitations. For the final presentation, "wake up" the services beforehand to ensure a smooth demo. |

---
## **HIGH-LEVEL PROJECT PLAN**

### **Project Roadmap (Revised)**

**Stage 1-2: Foundation & Planning (Weeks 1-4) ✅ COMPLETED**
* Initial model (v2.3) developed and validated at **52.5% accuracy** (40-match test).
* Project Charter drafted and key architectural decisions (PostgreSQL, FastAPI, React) made.

**Stage 3: Advanced Model Development (Weeks 5-6) 🔄 CURRENT**
* **Key Milestone:** Develop and validate the **specialized two-forest cascade model**.
* Engineer draw-specific features (`team_parity_score`, `elo_variance`).
* Optimize and freeze the final model architecture.

**Stage 4: Backend & Database Implementation (Weeks 7-8)**
* **Key Milestone:** A functional, deployed **PostgreSQL database and FastAPI backend**.
* Set up the database schema and populate it with historical data.
* Implement API endpoints to serve data and predictions.
* Develop and schedule the automated weekly data pipeline.

**Stage 5: Frontend Development (Weeks 9-10)**
* **Key Milestone:** An interactive **React dashboard** connected to the live backend.
* Build UI components for displaying performance metrics and future predictions.
* Implement visualizations (e.g., performance charts).

**Stage 6: Integration, Testing & Closure (Weeks 11-12)**
* **Key Milestone:** A fully integrated and tested **full-stack application**.
* End-to-end testing of the entire data flow.
* Final documentation and project presentation.

---
## **SUCCESS CRITERIA**

**Technical Success:**
* Final cascade model achieves **>52% accuracy** and **>30% draw recall** on the ongoing 2025-26 season.
* The automated weekly pipeline runs successfully with <5% manual intervention.
* The React frontend successfully fetches and displays live data from the FastAPI backend.

**Project Management Success:**
* Each stage of the roadmap is completed within its 2-week timeframe.
* The final application is successfully demoed at the end of Week 12.