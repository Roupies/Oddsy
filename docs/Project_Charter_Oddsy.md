# Oddsy Project Charter

**Project Name:** Oddsy - Premier League Match Prediction System  
**Project Manager:** Maxime Naguet  
**Date:** September 15, 2025  
**Version:** 1.0  

---

## 1. Project Objectives

### Purpose
To create an intelligent Premier League match prediction system that leverages advanced machine learning algorithms to predict football match outcomes (Home/Draw/Away) with superior accuracy compared to traditional baselines. The system will provide accessible predictions through a professional web application interface.

### SMART Objectives

1. **Achieve Superior Prediction Accuracy**: Maintain and validate a machine learning model with >50% accuracy in predicting Premier League match outcomes, significantly outperforming naive baselines (43.6% majority class, 33.3% random).

2. **Develop Production-Ready Web Application**: Create a responsive web application that allows users to view live match predictions, historical performance data, and model insights with 99.9% uptime and <2-second load times.

3. **Establish Robust Data Pipeline**: Implement an automated data acquisition and processing system that maintains 100% feature coverage for EPL 2025-26 season matches with real-time xG data integration.

---

## 2. Stakeholders and Team Roles

### Internal Stakeholders
- **Maxime Naguet** - Project Manager, Lead Developer, Data Scientist, QA Engineer

### External Stakeholders
- **End Users**: Football enthusiasts, betting analysts, sports data consumers
- **Data Providers**: UnderstatAPI, TheSportsDB, Premier League official sources
- **Academic Supervisors**: Course instructors and evaluators
- **Potential Investors/Partners**: Future business stakeholders interested in sports analytics

### Team Roles and Responsibilities

| Role | Responsibilities | Individual |
|------|-----------------|------------|
| **Project Manager** | Project planning, timeline management, risk assessment, stakeholder communication | Maxime Naguet |
| **Data Scientist** | Model development, feature engineering, performance validation, statistical analysis | Maxime Naguet |
| **Backend Developer** | API development, data pipeline architecture, model deployment, server management | Maxime Naguet |
| **Frontend Developer** | Web application UI/UX, responsive design, user experience optimization | Maxime Naguet |
| **DevOps Engineer** | Deployment automation, monitoring, infrastructure management | Maxime Naguet |
| **QA Engineer** | Testing, validation, performance monitoring, bug tracking | Maxime Naguet |

---

## 3. Project Scope

### In-Scope
- **Machine Learning Model**: Production-ready v2.3 model with 52.11% validated accuracy
- **Web Application**: Responsive frontend with prediction interface and historical data visualization
- **Data Pipeline**: Automated EPL 2025-26 data acquisition and processing system
- **Prediction System**: Real-time match predictions for upcoming EPL fixtures
- **Performance Monitoring**: Model accuracy tracking and validation infrastructure
- **Documentation**: Technical documentation, API documentation, user guides
- **Quality Assurance**: Comprehensive testing suite and validation pipelines

### Out-of-Scope
- **Multiple League Support**: Focus exclusively on Premier League (no Championship, other leagues)
- **Mobile Applications**: Web-only interface (no native iOS/Android apps)
- **Real-Time Betting Integration**: No direct betting platform APIs or financial transactions
- **Player-Level Analytics**: Team-level predictions only (no individual player performance)
- **Advanced Financial Features**: No investment tracking, portfolio management, or complex financial analytics
- **Multi-Language Support**: English interface only
- **Social Features**: No user accounts, comments, or social sharing functionality

---

## 4. Risk Assessment

| Risk Category | Risk Description | Probability | Impact | Mitigation Strategy |
|---------------|------------------|-------------|---------|-------------------|
| **Technical** | Model performance degradation with new season data | Medium | High | Continuous monitoring with audit_pipeline.py, weekly performance validation, automated retraining triggers |
| **Data** | Loss of access to UnderstatAPI or data source changes | Medium | High | Implement multiple data source backup strategies, local data caching, contract diversification |
| **Timeline** | Development delays due to single-person workload | High | Medium | Realistic timeline planning, MVP-first approach, regular milestone checkpoints |
| **Performance** | Web application scalability issues under load | Low | Medium | Performance testing, cloud hosting with auto-scaling, CDN implementation |
| **Quality** | Model validation failures or accuracy regression | Low | High | Rigorous audit pipeline, temporal validation, multiple seed testing |
| **External** | Changes in Premier League data availability or format | Medium | Medium | Flexible data ingestion architecture, format adaptation capabilities |

---

## 5. High-Level Project Plan

### Phase Overview

| Phase | Duration | Key Deliverables | Milestones |
|-------|----------|------------------|------------|
| **Stage 1: Foundation** | Week 1-2 | Team formation, project conceptualization | ✅ Completed |
| **Stage 2: Project Charter** | Week 3 | Project Charter document, stakeholder alignment | 🔄 Current |
| **Stage 3: Technical Documentation** | Week 4-5 | API specifications, architecture design, database schema | Technical blueprint complete |
| **Stage 4: MVP Development** | Week 6-10 | Web application, prediction API, data pipeline | Functional MVP deployed |
| **Stage 5: Testing & Optimization** | Week 11-12 | Performance testing, UI/UX refinement, deployment | Production-ready system |
| **Stage 6: Project Closure** | Week 13 | Final presentation, documentation, handover | Project complete |

### Detailed Timeline

#### Stage 3: Technical Documentation (Week 4-5)
- **Week 4**: API design, database schema, system architecture
- **Week 5**: Frontend wireframes, deployment strategy, testing framework

#### Stage 4: MVP Development (Week 6-10)
- **Week 6-7**: Backend API development, model integration
- **Week 8-9**: Frontend development, prediction interface
- **Week 10**: Integration testing, deployment pipeline

#### Stage 5: Testing & Optimization (Week 11-12)
- **Week 11**: Performance testing, bug fixes, optimization
- **Week 12**: User acceptance testing, final refinements

#### Stage 6: Project Closure (Week 13)
- **Week 13**: Final presentation preparation, documentation finalization

### Key Milestones
- ✅ **M1**: Project Charter Approved (Week 3)
- 🎯 **M2**: Technical Architecture Finalized (Week 5)
- 🎯 **M3**: MVP Beta Release (Week 10)
- 🎯 **M4**: Production Deployment (Week 12)
- 🎯 **M5**: Project Presentation (Week 13)

---

## 6. Success Criteria

### Technical Success Metrics
- Model accuracy maintains >50% on cross-validation
- Web application achieves <2-second load times
- 99.9% system uptime during evaluation period
- 100% feature coverage for EPL 2025-26 predictions

### Business Success Metrics
- Demonstrates clear value proposition over existing solutions
- Receives positive feedback from academic evaluators
- Shows potential for commercial viability
- Validates MVP concept for future development

### Quality Assurance Standards
- All code passes comprehensive audit pipeline
- 100% test coverage for critical prediction functions
- Professional documentation standards maintained
- Secure and scalable architecture implemented

---

## 7. Resource Requirements

### Technology Stack
- **Backend**: Python, FastAPI, scikit-learn, pandas
- **Frontend**: HTML5, CSS3, JavaScript (React.js)
- **Database**: PostgreSQL or SQLite
- **Deployment**: Cloud hosting (AWS/DigitalOcean)
- **Monitoring**: Application performance monitoring tools

### Development Environment
- **Hardware**: Current development machine sufficient
- **Software**: VS Code, Git, Docker, testing frameworks
- **Data Sources**: UnderstatAPI, TheSportsDB, EPL official data

---

## 8. Communication Plan

### Internal Communication
- **Daily**: Personal progress tracking and task management
- **Weekly**: Milestone review and risk assessment
- **Bi-weekly**: Stakeholder updates and documentation reviews

### External Communication
- **Weekly**: Progress reports to academic supervisors
- **Monthly**: Status updates to potential external stakeholders
- **Ad-hoc**: Issue escalation and support requests

---

## 9. Quality Management

### Validation Standards
- Continuous model performance monitoring via audit_pipeline.py
- Temporal validation maintaining TimeSeriesSplit methodology
- Code quality standards with comprehensive testing
- Documentation standards ensuring professional delivery

### Review Process
- Weekly self-assessment against project objectives
- Milestone-based quality gate reviews
- Continuous integration and deployment validation

---

**Document Prepared By:** Maxime Naguet  
**Approval Date:** September 15, 2025  
**Next Review:** September 22, 2025  

---

*This Project Charter serves as the foundational document for the Oddsy Premier League prediction system, establishing clear objectives, scope, and success criteria for the MVP development phase.*