# UID Matcher Enhanced

A comprehensive UID assignment system for survey questions that matches SurveyMonkey survey data with Snowflake reference questions and assigns unique identifiers (UIDs) using advanced machine learning algorithms.

## Overview

**Purpose**: Match survey questions from SurveyMonkey with a reference question bank in Snowflake and assign UIDs using ML algorithms including TF-IDF vectorization and semantic similarity.

**Key Features**:
- 🔗 **SurveyMonkey API Integration**: Extract surveys and questions with IDs
- ❄️ **Snowflake Database Connectivity**: Reference question bank with authority counts
- 🤖 **Machine Learning Matching**: TF-IDF vectorization + semantic similarity using sentence transformers
- 📊 **AMI Structure Categorization**: Survey Stages, Respondent Types, Programmes
- 🎯 **UID Final Reference System**: Authoritative UID assignments
- 🔍 **Identity vs Non-Identity Classification**: Automatic detection and separation
- 📤 **Export Functionality**: Prepared data upload back to Snowflake

## Application Versions

### Streamlit Version (Legacy)
- **File**: `uid_promax10.py` (~3160 lines)
- **Status**: Original monolithic implementation
- **Features**: Full functionality in single file

### Reflex Version (Current)
- **Basic Version**: `reflex_app.py` - Core functionality with modern UI
- **Enhanced Version**: `reflex_app_enhanced.py` - Complete implementation with advanced features
- **Modular Architecture**: Separated into specialized modules

## Architecture

### Core Modules

#### `reflex_app_enhanced.py` - Main Application
- **AppState**: Reactive state management with real-time updates
- **UI Components**: Modern component-based interface
- **Page Routing**: Multi-page application with navigation
- **Loading States**: Progress indicators and status messages

#### `database.py` - DatabaseManager
- **SurveyMonkey API**: Survey and question extraction with retry logic
- **Snowflake Connectivity**: Question bank queries with authority counts
- **Connection Management**: Automatic retry and error handling
- **Data Upload**: Batch operations for large datasets

#### `utils.py` - Utility Functions
- **Text Processing**: Enhanced normalization with synonym mapping
- **UID Matching Algorithms**: TF-IDF + semantic similarity
- **AMI Categorization**: Survey Stage/Respondent Type/Programme classification
- **Identity Detection**: Automatic PII identification and classification
- **Export Preparation**: Data formatting for Snowflake upload

#### `config.py` - Configuration Management
- **Environment Variables**: Secure credential management
- **API Configuration**: Token and connection string management
- **System Settings**: Thresholds and algorithm parameters

## Setup Instructions

### Prerequisites
- Python 3.8+
- SurveyMonkey API access token
- Snowflake database credentials

### Environment Configuration
Create a `.env` file with:
```env
SURVEYMONKEY_ACCESS_TOKEN=your_token_here
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_ROLE=your_role
```

### Installation
```bash
# Clone repository
git clone <repository_url>
cd UID_FINAL

# Install dependencies
pip install -r requirements.txt

# Initialize Reflex application
reflex init

# Run the application
reflex run
```

## Application Pages

### 🏠 Home Dashboard
- Connection status indicators
- System metrics and health checks
- Quick navigation to key features
- UID Final reference overview

### 📋 Survey Selection
- Browse and select SurveyMonkey surveys
- Extract questions with question IDs
- Preview survey data and schema types
- Batch processing with progress tracking

### 📊 AMI Categories
- Survey Stage classification (Recruitment, Pre-Programme, etc.)
- Respondent Type grouping (Participant, Business, Team member, etc.)
- Programme alignment (GYB, MEA, SYB, LDP, etc.)
- Unique question extraction by category

### 🔧 UID Matching
- Advanced ML-based question matching
- TF-IDF vectorization with configurable thresholds
- Semantic similarity using sentence transformers
- Manual UID assignment and configuration
- Match confidence indicators

### 📚 Question Bank
- Enhanced Snowflake question bank viewer
- UID Final reference integration
- Authority count analysis
- Unique UID table generation
- Conflict detection and resolution

### 🏗️ Survey Creation
- Design surveys using standardized questions
- Question validation against reference bank
- Schema type configuration
- Export ready for SurveyMonkey deployment

## Data Processing Pipeline

### 1. Data Extraction
- **SurveyMonkey**: API calls to extract surveys, questions, and choices
- **Caching**: Local caching to minimize API calls
- **Batch Processing**: Efficient handling of multiple surveys

### 2. Text Normalization
- **Enhanced Normalization**: Synonym mapping and stop word removal
- **Quality Scoring**: Algorithm to rank question quality
- **Deduplication**: Remove duplicate questions across surveys

### 3. AMI Structure Classification
- **Pattern Matching**: Keyword-based categorization
- **Survey Stages**: Automatic detection of survey lifecycle stage
- **Respondent Types**: Classification by participant role
- **Programmes**: Alignment with AMI programme structure

### 4. UID Matching
- **TF-IDF Vectorization**: Statistical text similarity
- **Semantic Matching**: Context-aware similarity using transformers
- **Authority Weighting**: Prioritize high-authority reference questions
- **Confidence Scoring**: Multi-level match confidence

### 5. Identity Classification
- **PII Detection**: Automatic identification of personal information
- **Identity Types**: Granular classification (name, email, phone, etc.)
- **Separation**: Split into identity and non-identity datasets

### 6. Export Preparation
- **Table Generation**: Separate tables for identity and non-identity questions
- **Snowflake Ready**: Formatted for database upload
- **CSV Export**: Downloadable formats for analysis

## Key Improvements (Reflex vs Streamlit)

### Performance
- **Reactive State**: Real-time updates without page refreshes
- **Async Operations**: Non-blocking database calls and API requests
- **Optimized Rendering**: Component-based UI with efficient updates

### Architecture
- **Modular Design**: Separated concerns into specialized modules
- **Error Handling**: Comprehensive try-catch with user-friendly messages
- **Configuration Management**: Environment-based secure credential handling

### User Experience
- **Modern UI**: Component-based interface with loading states
- **Progress Indicators**: Real-time feedback during long operations
- **Status Messages**: Clear communication of system state
- **Responsive Design**: Optimized for different screen sizes

### Maintenance
- **Code Organization**: Clean separation of business logic and UI
- **Type Safety**: Enhanced type hints and validation
- **Documentation**: Comprehensive inline documentation
- **Testing Ready**: Structure supports unit and integration testing

## Configuration

### Matching Thresholds
```python
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
```

### AMI Structure Categories
- **Survey Stages**: 12 categories from Recruitment to Longitudinal
- **Respondent Types**: 6 types from Participant to Managers
- **Programmes**: 11 programmes including GYB, MEA, SYB, LDP

### UID Final Reference
- **548 authoritative mappings** from question text to UID Final
- **Direct lookup**: Fast exact matching
- **Fuzzy matching**: Handles minor text variations

## Troubleshooting

### Common Issues
1. **Connection Failures**: Check environment variables and network connectivity
2. **Rate Limiting**: Built-in retry logic with exponential backoff
3. **Memory Issues**: Batch processing for large datasets
4. **UID Conflicts**: Authority count resolution and manual override options

### Support
- Check logs for detailed error messages
- Verify credentials and API access
- Review network connectivity for external services
- Use the question bank validation for standardization

## Migration Notes

### From Streamlit to Reflex
- **State Management**: `st.session_state` → `rx.State` with reactive properties
- **UI Components**: `st.sidebar` → custom navigation components
- **Data Display**: `st.dataframe()` → custom table components with editing
- **Forms**: `st.form()` → `rx.form()` with reactive validation
- **File Operations**: Server-based → browser-based with custom handlers

### Preserved Functionality
All original features maintained:
- ✅ Survey selection and question extraction
- ✅ AMI structure categorization  
- ✅ Advanced UID matching algorithms
- ✅ Question bank management
- ✅ Identity classification
- ✅ Export capabilities
- ✅ Snowflake integration

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]