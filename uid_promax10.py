# ============= IMPORTS AND DEPENDENCIES =============
import streamlit as st
import pandas as pd
import requests
import re
import logging
import json
import time
import os
import numpy as np
from uuid import uuid4
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from collections import defaultdict, Counter

# ============= STREAMLIT CONFIGURATION =============
st.set_page_config(
    page_title="UID Matcher Enhanced", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# ============= LOGGING SETUP =============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= CONSTANTS AND CONFIGURATION =============

# Matching thresholds
TFIDF_HIGH_CONFIDENCE = 0.60
TFIDF_LOW_CONFIDENCE = 0.50
SEMANTIC_THRESHOLD = 0.60
HEADING_TFIDF_THRESHOLD = 0.55
HEADING_SEMANTIC_THRESHOLD = 0.65
HEADING_LENGTH_THRESHOLD = 50

# Model and API settings
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 1000
CACHE_FILE = "survey_cache.json"
REQUEST_DELAY = 0.5
MAX_SURVEYS_PER_BATCH = 10

# Identity types for export filtering
IDENTITY_TYPES = [
    'full name', 'first name', 'last name', 'e-mail', 'company', 'gender', 
    'country', 'age', 'title', 'role', 'phone number', 'location', 
    'pin', 'passport', 'date of birth', 'uct', 'student number',
    'department', 'region', 'city', 'id number', 'marital status',
    'education level', 'english proficiency', 'email', 'surname',
    'name', 'contact', 'address', 'mobile', 'telephone', 'qualification',
    'degree', 'identification', 'birth', 'married', 'single', 'language',
    'sex', 'position', 'job', 'organization', 'organisation'
]

# Enhanced synonym mapping
ENHANCED_SYNONYM_MAP = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
    "how many people report to you": "team size",
    "how many staff report to you": "team size",
    "what is age": "what is your age",
    "what age": "what is your age",
    "your age": "what is your age",
    "current role": "current position",
    "your role": "your position",
}

# AMI Structure Categories
SURVEY_STAGES = {
    "Recruitment Survey": ["application", "apply", "applying", "candidate", "candidacy", "admission", "enrolment", "enrollment", "combined app"],
    "Pre-Programme Survey": ["pre programme", "pre-programme", "pre program", "pre-program", "before programme", "preparation", "prep"],
    "LL Feedback Survey": ["ll feedback", "learning lab", "in-person", "multilingual"],
    "Pulse Check Survey": ["pulse", "check-in", "checkin", "pulse check"],
    "Progress Review Survey": ["progress", "review", "assessment", "evaluation", "mid-point", "checkpoint", "interim"],
    "Growth Goal Reflection": ["growth goal", "post-ll", "reflection"],
    "AP Survey": ["ap survey", "accountability partner", "ap post"],
    "Longitudinal Survey": ["longitudinal", "impact", "annual impact"],
    "CEO/Client Lead Survey": ["ceo", "client lead", "clientlead"],
    "Change Challenge Survey": ["change challenge"],
    "Organisational Practices Survey": ["organisational practices", "organizational practices"],
    "Post-bootcamp Feedback Survey": ["post bootcamp", "bootcamp feedback"],
    "Set your goal post LL": ["set your goal", "post ll"],
    "Other": ["drop-out", "attrition", "finance link", "mentorship application"]
}

RESPONDENT_TYPES = {
    "Participant": ["participant", "learner", "student", "individual", "person"],
    "Business": ["business", "enterprise", "company", "entrepreneur", "owner"],
    "Team member": ["team member", "staff", "employee", "worker"],
    "Accountability Partner": ["accountability partner", "ap", "manager", "supervisor"],
    "Client Lead": ["client lead", "ceo", "executive", "leadership"],
    "Managers": ["managers", "management", "supervisor"]
}

PROGRAMMES = {
    "Grow Your Business (GYB)": ["gyb", "grow your business", "grow business"],
    "Micro Enterprise Accelerator (MEA)": ["mea", "micro enterprise", "accelerator"],
    "Start your Business (SYB)": ["syb", "start your business", "start business"],
    "Leadership Development Programme (LDP)": ["ldp", "leadership development", "leadership"],
    "Management Development Programme (MDP)": ["mdp", "management development", "management"],
    "Thrive at Work (T@W)": ["taw", "thrive at work", "thrive", "t@w"],
    "Bootcamp": ["bootcamp", "boot camp", "survival bootcamp", "work readiness", "get set up"],
    "Academy": ["academy", "care academy"],
    "Finance Link": ["finance link"],
    "Custom": ["winning behaviours", "custom", "learning needs"],
    "ALL": ["all programmes", "template", "multilingual"]
}

# UID governance rules
UID_GOVERNANCE = {
    'max_variations_per_uid': 50,
    'semantic_similarity_threshold': 0.85,
    'auto_consolidate_threshold': 0.92,
    'quality_score_threshold': 5.0,
    'conflict_detection_enabled': True,
    'conflict_resolution_threshold': 10,
    'high_conflict_threshold': 100
}

# Heading reference texts
HEADING_REFERENCES = [
    "As we prepare to implement our programme in your company, we would like to define what learning interventions are needed to help you achieve your strategic objectives.",
    "Now, we'd like to find out a little bit about your company's learning initiatives and how well aligned they are to your strategic objectives.",
    "This section contains the heart of what we would like you to tell us. The following twenty Winning Behaviours represent what managers and staff do in any successful and growing organisation.",
    "Welcome to the Business Development Service Provider (BDSP) Diagnostic Tool, a crucial component in our mission to map and enhance the BDS landscape in Rwanda.",
    "Thank you for dedicating your time and effort to complete this diagnostic tool. Your valuable insights are crucial in our mission to map the landscape of BDS provision in Rwanda.",
    "Clearly establishing these parameters now will help to inform and support the embedding of a culture of learning and personal development from your leaders all the way through to your non-management staff.",
    "Please provide the following details:",
    "A learning initiative is any formal or informal way in which your people are encouraged to learn. It may include online or face-to-face courses, coaching, projects, etc.",
    "Most or all of them would be important for your people too, but we are interested in identifying those few that stand out as supporting your unique strategic opportunities and challenges - those that would make a significant and obvious difference to your organisation's performance.",
    "Now, we want to delve a bit deeper to examine how the winning behaviours you have prioritised at the company-wide level might look different if you just focused on those employees who manage people.",
    "Now, we want to delve a bit deeper to examine how the winning behaviours you have prioritised at the company-wide and manager levels might look different if you just focused on those employees who do not manage any people.",
    "As a last step, we ask that you rank order the short list of those you have indicated are both important and less frequent for non-managers.",
    "Thank you for taking the time to reflect on how aligned your learning initiatives are with your key strategic priorities.",
    "Please provide the followng details:",
    "BUSINESS DETAILS",
    "LEARNING NEEDS",
    "Confidentiality Assurance",
    "The information provided in this assessment will be treated with the utmost confidentiality and will be used solely for the purpose of developing and improving our access to finance programs.",
    "Contact Information",
    "Institutional Profile",
    "Section II: Financial Products and Services",
    "Section III: Credit Assessment and Risk Management",
    "Introduction",
    "This survey is designed to capture vital information about your organization's profile, services, target market, business model, and impact measurement practices.",
    "Your participation in this 15-20 minute survey is invaluable in shaping the future of BDS in Rwanda.",
    "Confidentiality",
    "Contact Information",
    "Organizational Profile and Reach",
    "Service Offering and Delivery",
    "Target Market and Specialization",
    "Understanding your target market helps us identify any underserved segments and opportunities for expanding BDS reach.",
    "Business Model and Sustainability",
    "This section assesses your organization's financial health and sustainability, helping us identify areas where BDS providers might need support.",
    "Understanding how you measure impact and the challenges you face helps us develop better support systems and identify areas for improvement in the BDS ecosystem.",
    "Ecosystem Collaboration and Support",
    "This section explores how BDS providers interact within the larger ecosystem and what support would be most beneficial.",
    "Future Outlook",
    "Understanding your future plans and perspectives helps us anticipate trends and prepare for the evolving needs of the BDS sector.",
    "Thank You for Your Participation",
    "Your participation is a significant step towards creating a more robust, responsive, and effective BDS ecosystem that can drive sustainable MSME growth and contribute to Rwanda's economic development.",
    "MSME Survey Tool: Understanding BDS Provision in Rwanda",
    "Section 1: Contact Information",
    "Section 2: Business Challenges and BDS Engagement",
    "Section 3: BDS Quality and Needs Assessment",
    "Section 4: Future Engagement",
    "Conclusion",
    "Please fill in the details below:",
    "Your organisation's current learning initiatives",
    "RATE: Pinpoint your organisation's key Winning Behaviours",
    "FREQUENCY: Tell us about how often the Winning Behaviours are displayed in your organisation",
    "First, we need to know a bit about your role",
    "RATE: Pinpoint key Winning Behaviours for managers",
    "FREQUENCY: Tell us how often the Winning Behaviours are displayed by managers",
    "RANK: Prioritise the Winning Behaviours to focus on for managers",
    "As a last step, we ask that you rank order the short list of those you have indicated are both important and less frequent for managers.",
    "RATE: Pinpoint key Winning Behaviours for non- managers",
    "FREQUENCY: Tell us how often the Winning Behaviours are displayed by non-managers",
    "RANK: Prioritise the Winning Behaviours to focus on for non-managers",
    "Please describe in very practical behaviours what your people need to do differently to achieve your strategic goals.",
    "Prioritise the Winning Behaviours to focus on",
    "Ecosystem Support Organizations Interview Guide",
    "Introduction (5 minutes)",
    "Introduce yourself and the purpose of the interview",
    "Assure confidentiality and ask for permission to record the interview",
    "Explain that the interview will take about 1 hour",
    "Target Market and Specialization"
]

# ============= UID FINAL REFERENCE DATA =============
UID_FINAL_REFERENCE = {
    "On a scale of 0-10, how likely is it that you would recommend AMI to someone (a colleague, friend or other business?)": 1,
    "Do you (in general) feel more confident about your ability to raise capital for your business?": 38,
    "Have you set and shared your Growth Goal with AMI?": 57,
    "Have you observed an improvement in the following areas in your business since engaging with AMI?": 77,
    "If jobs were created, please specify how many:": 78,
    "As well as your core business KPI, what is your social and/or environmental KPI and did that increase? (If you do not have a social and/or environmental KPI, please indicate \"N/A\" below.)": 80,
    "How would you report your overall happiness at work?": 88,
    "Have you noticed a positive behaviour change in this participant since they engaged with CALA?": 119,
    "Have you noticed a positive behaviour change in this participant since they engaged with AMI?": 119,
    "Where does your company do business? (Please tick as many as apply)": 124,
    "Is your business based in an urban or rural area?": 125,
    "What sector do you work in?": 126,
    "Is your business formally registered?": 127,
    "Is your business driven by a social mission, a desire to protect the environment or both?": 128,
    "At the end of last year, how many employees did you have?": 130,
    "Has your business obtained a loan in the last 12 months?": 131,
    "Did you sell any shares last year?": 132,
    "Are you looking for external finance this year ?": 133,
    "Where did you hear about this programme?": 134,
    "Which trajectory best describes where your business is at?": 136,
    "Did the facilitator seem knowledgeable, well-prepared and communicate effectively?": 151,
    "What specific tools and resources have you found most useful in this phase?": 156,
    "How useful did you find the tools and resources in this phase? (On a scale of 1-7 where 1 is not useful at all and 7 is extremely useful)": 156,
    "Is the content of this programme pitched at the appropriate level?": 157,
    "Is your learning journey clear to you?": 158,
    "Please indicate the value of loans taken this year. Use the local currency that you selected for your financial reporting. If you did not take out a loan please type '0'": 159,
    "Please indicate the value of shares sold.Use the local currency that you selected for your financial reporting. If you did not sell shares  type '0'": 160,
    "Can you attribute any of this performance improvement to AMI?": 161,
    "Since your organisation engaged with AMI, have you noticed an improvement in any of the following with regards to the participants' performance? (Tick as many as apply)": 162,
    "Do you also feel that there were improvements on any of the following people metrics in your company / unit since working with AMI?": 164,
    "Since engaging with AMI, have you noticed an improvement in any of the following with regards to your own personal progress? (Tick as many as apply):": 170,
    "Are you still applying what you learned on the AMI programme in your business?": 171,
    "Have you improved in any of the below areas since participating in the programme?": 173,
    "As you reflect on the last 6 months, do you feel that the Grow Your Business programme achieved what you hoped it would?": 185,
    "How likely are you to engage with AMI again?": 186,
    "Is a promotion and/or pay rise for staff who participated in AMI's programme likely?(On a scale of 1-7 where 1 is Not At All Likely and 7 is Extremely Likely)": 187,
    "Are these opportunities for promotion/pay rise more likely for your male or female staff (who participated in this programme)?": 188,
    "Has the retention of staff managed by those who participated in the programme increased after the programme?": 189,
    "Have you noticed any difference in the retention of male and female staff?": 190,
    "Did you track your business' performance against your Growth Goal over the past 3 months?": 191,
    "Did your business achieve your revenue target last month?": 192,
    "Please tell us more - what have you learned from this? Will you do anything differently this month?": 193,
    "We'd love to find out more about how your Change Challenge is going:": 194,
    "What has been most useful to you in this phase? Please select the ones that have been most useful - maximum of 3.": 195,
    "What could be improved or what is missing from this programme? This could relate to the online content, virtual sessions, pod groups, learning journey or additional support that you need eg. Access to finance / a broader network of entrepreneurs / mentorship?": 196,
    "Please evaluate your AMI experience compared to other business training or education programmes you may have taken.": 197,
    "How has this programme helped you? Describe what has changed since you participated in the programme.": 198,
    "What key insight / learning are you taking away from this session?": 199,
    "We'd love to find out more about how your ALP is going:": 200,
    "Have you made progress on your business goals since beginning your Grow Your Business journey?": 204,
    "What kind of support do you want in the ongoing Growth Network for Entrepreneurs to help you over the next 6-12 months? (Please select all that you are interested in.)": 205,
    "Given your business' performance last month, which talent practice have you identified to support you in achieving the next step towards your Growth Goal? (We recommend selecting one practice to focus on this phase but you are able to select up to 2.)": 206,
    "Do you need any support in tracking your progress towards your Growth Goal?": 207,
    "What are your business' 2 biggest struggles right now? (Please select the top 2.)": 209,
    "Where do you think you'll do most of your online learning?": 210,
    "What is your current employment status?": 211,
    "How likely are you to start a business within the next 12 months? (On a scale of 1-7 where 1 is highly unlikely and 7 is highly likely)": 212,
    "Which lesson in the Entrepreneurship: From Idea to Action course have you found most useful so far?": 213,
    "What has been your biggest blocker in terms of working through the online course?": 214,
    "What have been the most significant outcomes from the programme for you?": 215,
    "Do you feel more confident about your ability to lead your business towards growth?": 226,
    "Other than the Rise sessions, is there anything else that AMI can offer to support you at this time?": 227,
    "Did you achieve your Growth Goal for last year?": 229,
    "Do you feel more confident about launching your business idea / driving your business forward?": 230,
    "What is your gender?": 233,
    "What is your age?": 234,
    "What did you find most useful about today's session?": 236,
    "What do you hope to get out of the next 4 weeks of access to AMI's online platform?": 237,
    "How can we help you to get started?": 238,
    "What was your business' annual revenue for 2019?": 240,
    "Which tool/s did you download and use?": 241,
    "How did you use this tool/s? What was the result / impact of using it?": 242,
    "Please provide the following details:": 243,
    "As you reflect on your Change Challenge, what changes / impact / results have you seen? Please be as specific as possible. Were there cost savings? Did it help the business? Has your team been able to do more? Are people happier?": 244,
    "Review Strategy: We have had an annual strategy review discussion for this year (~P2~)": 245,
    "Track Progress: Each person responsible measures and tracks progress against the plan (~P3~)": 246,
    "Review Business Performance: We meet quarterly to review progress against strategic objectives and set targets for the next quarter (~P4~)": 247,
    "Plan Operations: We anticipate what people and resources will be needed, delegate and coordinate these across the company and procure and allocate resources in good time (~P5~)": 248,
    "Compliance: We review annually that we are compliant with all necessary laws and regulations, have appropriate internal policies, and have prepared for possible risks (~P6~)": 249,
    "Survey Customers: We regularly ask customers (existing, potential and former) if our products/services are the right fit for them (~C1~)": 250,
    "Survey Competitors: We regularly identify and look at our competitors' products, prices & service levels to understand how we compare (~C2~)": 251,
    "Survey Suppliers: We regularly ask suppliers which products / services are selling well in this sector to update our offering (~C3~)": 252,
    "Innovate: We review our market research data and customer feedback to assess existing revenue models and find new sources of revenue (~C4~)": 253,
    "Implement a marketing and sales plan: We regularly identify sales and marketing activities to promote our products and increase sales (~C5~)": 254,
    "Capture all money transactions: We keep records of all our business transactions, including sales, purchases, loans, deposits and payments (~M1~)": 255,
    "Review financial performance: We review weekly/monthly how the business is doing financially (eg. Revenue / costs / profits / cashflow) (~M2~)": 256,
    "Budget: Each year we predict what our income and costs will be each month and each month we compare actuals against what we expected and make decisions accordingly (~M3~)": 257,
    "Reduce costs: We regularly review what we spend money and time on and take steps to reduce what we spend (~M4~)": 258,
    "Negotiate with suppliers: We regularly negotiate with suppliers for a lower price, better payment terms, better quality and/or better service on material / services (~O1~)": 259,
    "Track and manage inventory: We regularly track inventory to monitor the movement of inventory and ensure that the business does not run out of stock (~O2~)": 260,
    "Follow clear operational processes: Staff follow our key business processes to ensure we deliver consistent quality and value (~O3~)": 261,
    "Streamline operations: At least monthly we get together with those involved to review our business processes to identify opportunities for improvement (~O4~)": 262,
    "Regular communication and check-in with staff: As business leaders, we regularly check-in with staff to align on direction, motivate and problem-solve(~T1~)": 263,
    "Manage organisational climate: We take a regular pulse check to measure the organisational climate (eg. Trust, job satisfaction, quality of interactions) (~T2~)": 264,
    "Regular feedback and coaching: At least once a month, we meet with staff members to give feedback and coaching (~T3~)": 265,
    "Performance reviews: Every staff member has a formal performance review every quarter (~T4~)": 266,
    "Staff learning: We dedicate the equivalent of at least half a day every month for people to learn what they need to do well in their jobs (~T5~)": 267,
    "Based on your self-assessment of your business against the 25 core business practices, what practice(s) have you identified to focus on next? Our goal is to identify the practice(s) and learning needs that will move you, and other businesses like you, closer to achieving your business goals. Please indicate your selected practice(s) below (you can select up to three).": 268,
    "Do you have a plan to get started with this practice, and keep progressing towards your Growth Goal?": 271,
    "Are you happy with your pod / peer group? Do you find your engagement with other participants valuable?": 272,
    "As you reflect on your coursework, which elements of the course did you find useful? Please tick all that apply.": 273,
    "On a scale of 0-10, how likely would you be to recommend the AMI's online courses below to someone (a colleague, friend or other business)?": 274,
    "What has been the most significant change to your business as a result of participating in the AMI programme?": 275,
    "What support from AMI would be most useful at this time?": 277,
    "What is the biggest challenge you're experiencing in your job right now?": 278,
    "Please tell us more about your entrepreneurial journey:": 279,
    "Can you describe your enterprise in 1-2 sentences?": 280,
    "What is the problem that your enterprise seeks to address?": 281,
    "What year did your enterprise start?": 282,
    "How many paid staff work in your business? (Please include yourself in this number)": 283,
    "Has your enterprise generated revenue?": 284,
    "What is your current financial model?": 285,
    "What do you want to learn to support you to drive your enterprise forward?": 286,
    "Can you describe your idea in 1-2 sentences?": 287,
    "What is the problem that your idea seeks to address?": 288,
    "What do you want to learn to support you to launch your enterprise?": 289,
    "What top 3 skills would you like to get out of this programme? Refer to the list below and select the three skills that you're most excited about developing.": 290,
    "As you begin this Grow Your Business programme, what are you most interested in / excited about?": 291,
    "What would be your success metric at the end of the programme i.e. how will you measure whether this programme has really impacted your business? Please try to be as specific as possible.": 292,
    "Are you currently participating in any other learning programme?": 293,
    "You indicated that you are currently participating / have applied to participate in another learning programme. Please tell us more. What does this programme focus on?": 294,
    "What time commitment is required from you as part of this programme? Eg. I attend 2-hour virtual workshops every month and work through 1 online course each month (which takes 2-3 hours)": 295,
    "Are you currently participating in any other business development / learning programme?": 296,
    "As we kick-off the programme, we'd like to support you as much as possible to get the most out of the programme. Is there anything you feel may limit your active participation in the programme?": 297,
    "How many people report to you?": 298,
    "What is your core objective for this programme - why are you here?": 299,
    "How comfortable are you learning online?": 300,
    "Do you prefer video, downloadable text, or audio-based resources?": 301,
    "This programme involves monthly 2-hour learning sessions where you'll have an opportunity to connect with fellow peers and explore content designed to help you grow. What time of day would suit you best for these monthly learning sessions? Please tick ALL that apply.(Please note that we cannot guarantee that your preference will be accommodated - we will review all responses to this question to identify the most popular timeslot for participants.)": 302,
    "Do you feel comfortable discussing topics like goal-setting, effective communication, handling stressful situations and empowering your team with your direct manager?": 304,
    "What is their relationship to you?": 305,
    "What additional support / guidance do you need to help you drive your ALP forward?": 306,
    "The Challenge: What is something you're worrying about... Something that isn't working currently & that you want to change at work / with your team?": 320,
    "Desired Situation: What would it look like if it was working?": 321,
    "Metrics: How will you be able to tell that you've succeeded at addressing this challenge? Please be as specific as possible.": 322,
    "Possible Root Causes: Why do you think this particular issue / challenge is happening? What might you or others be doing wrong? What beliefs / mindsets are driving this?": 323,
    "My Plan: How are you going to make time to work on your Change Challenge?": 324,
    "Do you need any help to get started with your Change Challenge?": 325,
    "Were you able to implement your Action Learning Project?": 326,
    "What impact has your Action Learning Project had / is likely to have on your team / organisation? Please select all that apply.": 327,
    "What has been the most difficult component of the Action Learning Project for you?": 328,
    "As you reflect on your Action Learning Project, what could be improved or what is missing? This could be related to the ALP tools, coaching sessions or guidelines related to various milestones etc.": 329,
    "Have you met the goals you set when you began this programme?": 330,
    "What support / opportunities would allow you to keep learning and growing as a manager and leader?": 331,
    "How did you find your facilitator throughout your journey with AMI?": 332,
    "What do I want to get out of this programme?": 333,
    "What is your job title/role in the business?// Ushinzwe iki muri ubwo bucuruzi?": 334,
    "What is your job title/role in the business?": 335,
    "Please tell us more about your response above (question 12).": 336,
    "What is the best way to contact you?": 337,
    "What can I contribute / what do I bring to this programme?": 338,
    "Do you feel that you have the support you need to get the most out of this programme? If yes, what support do you have available (eg. Dedicated learning time, accountability partner, device/internet access)? If no, what support would you wish for that you're currently missing)?": 339,
    "What did you love about the online courses? What did you wish was different? This could relate to a specific course or your general experience of working through AMI's online courses.": 340,
    "Are you more effective at work since participating in the programme?": 341,
    "After our first Learning Lab, you identified a \"Change Challenge\" - something practical that you wanted to improve in your team / at work. Please describe what you wanted to change and some of the things you've done to make the change happen.": 342,
    "What was the most valuable framework, tool or course that helped you address your Change Challenge?": 343,
    "Did you know what was covered in the programme in which the participant(s) you supported was participating?": 345,
    "Optional: Please describe in your own words how you have benefited from engaging with AMI.": 346,
    "Your email address:": 350,
    "Will you / have you split your Growth Goal over the 12 months of this year? (Please don't forget to make a note of this so that you're able to track your progress against the target for each month!)": 352,
    "Do you feel better equipped to address the challenges you've experienced in your role/department? (On a scale of 1-7 where 1 is Not At All and 7 is Extremely)": 353,
    "What do you feel are your remaining gaps? Tick all that apply.": 354,
    "How useful did you find the lessons you have completed?": 355,
    "What has been your biggest insight from the Entrepreneurship: From Idea to Action course so far?": 356,
    "What is your plan to get back on track with the Entrepreneurship: From Idea to Action course?": 357,
    "How is your online coursework going? Please select the option that best describes your progress.": 358,
    "How frequently have you applied the knowledge and skills gained from the porgamme in your day-to-day business operations?": 359,
    "Please select your country area code followed by your cellphone number with no spaces (eg. 794000000):": 360,
    "What is your Growth Goal for this year? You can indicate this as a specific number eg. KSH 750,000 OR as a percentage growth of last year's revenue eg. We want to grow our revenue by 15%.": 361,
    "You indicated that you have tracked your business' performance against your Growth Goal over the past 6 months.What have you learned from this process of tracking and reflecting on your business' performance against your Growth Goal target?": 362,
    "Have you downloaded the Preparing for Finance tool?": 363,
    "Do you know how to use this tool?": 364,
    "Given your specific planting, growing, and harvesting seasons, which are your busiest months in the year. ie. When would you find it most difficult to fully participate in this programme? Select ALL that may apply": 365,
    "When was the first year your business made a sale?": 367,
    "What is one specific outcome you hope to achieve for your business from this AMI programme? Please select one from the list below.": 368,
    "If your business is on track to graduate from the Grow Your Business programme, would you like to join AMI's pan-African Growth Network? (*Please note: This opportunity is exclusively available to graduating businesses.)": 369,
    "What kind of support are you most excited about as you look forward to joining AMI's Growth Network? Please select all that you are interested in.": 370,
    "Is your business still up and running since you joined the AMI learning programme?": 371,
    "Are you still involved in this business?": 372,
    "Are you still an entrepreneur ie. Have you started a new business since participating in your learning programme with AMI?": 373,
    "Has your company obtained philanthropic or grant funding in the last 12 months?": 374,
    "Does your business work with smallholder farmers?": 375,
    "How does your work influence smallholder farmers?": 376,
    "Do you identify as having a disability?": 377,
    "Do you identify as a refugee or displaced person?": 378,
    "Dose your salary / wage reliably meet your needs and that of your dependants?": 379,
    "Is your job well-regarded as reputable and honest by society at large?": 380,
    "Are you treated with respect, appreciation and dignity at work?": 381,
    "Do you have a sense of satisfaction, purpose and accomplishment at work?": 382,
    "Please select the currency used for your financial reporting:": 384,
    "Total Revenue for your 2021  financial year. (Your total sales for the year 2021) Please enter whole numbers only, for example, 10000. Do not use decimals or commas. Also, DO NOT add the currency abbreviation. ": 385,
    "Total Costs for your 2021 financial year. (Your total expenses for 2021)Please enter whole numbers only, for example, 10000. Do not use decimals or commas. Also, DO NOT add the currency abbreviation. ": 386,
    "Total Profit for your 2021 financial year * (Your total revenue less your total costs )Please enter whole numbers only, for example, 10000. Do not use decimals or commas. Also, DO NOT add the currency abbreviation.": 387,
    "What was the #1 thing that kept you focused and motivated as you worked towards completing the core milestones and graduating from this programme?": 391,
    "What practice/s have you implemented / started to implement that you're finding most impactful?": 392,
    "Which sections / parts of the tool do you feel comfortable with?": 393,
    "Which sections / part of the tool feel confusing or unclear?": 394,
    "Where can you see yourself getting blocked / stuck in an Access to Finance process? Please provide as much detail as possible - we'd love to help!": 395,
    "The following indicators are relevant to access to finance for your firm. Please select appropriately whether the following indicators have decreased, remained unchanged or increased over the past 6 months in your business?": 396,
    "Have these business priorities changed since the beginning of the programme? How have they changed and why?": 401,
    "What are your current Business Priorities? (E.g. finance, people)": 402,
    "Are you interested in the 3-month mentorship programme offered by Stanford Seed?": 403,
    "Date of Birth [DD/ MM/ YY]": 404,
    "What is something you're worrying about... What is the one thing you want to focus on?": 408,
    "What would it look like if it was working?": 409,
    "How will you be able to tell that you've succeeded?": 410,
    "Why do you think this particular issue / challenge is happening? What might you or others be doing wrong? What beliefs / mindsets are driving this?": 411,
    "What do you commit to doing to change this? What new tools / ideas will you try? When / how will you implement this? What is your plan?": 412,
    "Is setting your Growth Goal helping you to plan or think about your future better and more often?": 425,
    "How many years of managerial experience do you have?Input the number of years in figures.": 427,
    "Have you undertaken a similar business or management training before?": 428,
    "Have you attended any AMI programme in the past?": 429,
    "Are you happy for us to contact you for more information for our programme impact assessment and marketing purposes?": 430,
    "Have you implemented at least 1 practice in your business so far?": 431,
    "Describe any changes you've observed in your team or business based on the practice you implemented (and list the practice you are referring to). Please be as specific as possible.": 432,
    "Which tool(s) did you use to help you implement this practice/s?": 433,
    "In this phase, your core milestone is to work through the Bookkeeping Basics course. This course will showcase the key tools that we explored in Learning Lab 2, and will guide you through using these tools in your business. Which of the tools / concepts from Learning Lab 2 are you most interested in exploring?": 434,
    "How would you like to submit your Growth Goal?": 437,
    "You indicated that you have not tracked your business' performance against your Growth Goal over the past 3 months.How do you understand your business' monthly performance? What is your biggest blocker to regular tracking?": 439,
    "Reflecting on the SMT sessions we have had in the last year;": 440,
    "Given the several topics discussed on SMT in 2022;": 441,
    "Which topics would you like us to address this year?": 442,
    "Is there a specific topic you would be comfortable sharing with the rest of the team?": 443,
    "Please specify the topic..": 444,
    "Which month would you like to share the topic above?": 445,
    "What type of (non) finance provider are you?": 446,
    "What motivated you to enroll in the DGGF Academy?": 447,
    "What are you most excited about interacting with at the DGGF Academy? Please select the ones that you will find most useful - you'll be able to select a maximum of 3.": 448,
    "How do you envision benefiting from the Academy?": 449,
    "To gain insights into your past engagements, please indicate whether you have collaborated with any professionals from your sector.Kindly select the most fitting option from the list below that best characterizes the nature of your collaboration.": 450,
    "If there was one topic you wanted the DGGF Academy to cover in 2024, what would it be?": 451,
    "Are there any networking events/conferences that you plan on attending in 2024?": 452,
    "What major changes have you made to your business in the last year?": 453,
    "To what extent were these changes dependent on AMI's support?": 454,
    "What is the current stage of your business operation?": 455,
    "How familiar are you with the loan application process for MSMEs?": 456,
    "What types of financial challenges do you currently face in securing loans for your MSME?": 457,
    "How would you rate the responsiveness of banks and other financial providers in providing feedback on loan applications?": 458,
    "How would you rate the ease of communication with banks during the loan application process?": 458,
    "Are you aware of any bank loan programs or initiatives designed to support MSMEs?": 459,
    "What types of financial support do you believe would be most beneficial for your MSME? (Select up to three)": 460,
    "How satisfied are you with the current availability of loan products tailored for MSMEs?": 461,
    "What information or resources do you believe are lacking in supporting MSMEs in preparing loan applications?": 462,
    "What types of support or resources do you believe would be most beneficial in improving access to loans for your MSME?": 464,
    "What are the two most important business needs that you have right now ? (Please choose exactly two)": 465,
    "Have you ever faced challenges in understanding or meeting the eligibility criteria for MSME loans?": 466,
    "How confident are you in the ability of your MSME to meet the repayment obligations of a loan?": 467,
    "In which year did your business first begin offering Business Development Support (BDS) services?": 477,
    "How many SMEs does your BDS service support?": 478,
    "How would you describe your involvement in your business?": 479,
    "What form of business development support do you provide?": 480,
    "What is the approximate annual revenue size range of the SMEs that your business caters to?": 481,
    "On a scale of 1 to 10, how would you rate your current level of knowledge and experience in business development? (1 = very low, 10 = very high)": 483,
    "What topics or areas of business development are you most interested in learning about during the program?": 484,
    "We would like to understand what level of management you are in. Select the answer that most closely matches your current management role.": 486,
    "What functional area of the organisation are you a part of?": 487,
    "Please review the following 20 winning behaviours and rate how important each is to achieving your company's strategic objectives.": 488,
    "Please rate how frequently you display the following winning behaviours:": 489,
    "What are the strategic company objectives that are most important right now?(Please enter at least one objective.)": 490,
    "What learning initiatives do you currently have in your organisation? If none, write N/A.": 491,
    "To what extent are your current learning initiatives aligned with your company's strategic objectives?": 491,
    "Please review the following 20 winning behaviours in relation to the team that you manage and rate how important each is to achieving your organisation's strategic objectives.": 492,
    "You selected the Winning Behaviours below as either significantly or most importantly contributing to achieving your organisation's strategic objectives. Now, we'd like to understand about how frequently these are displayed by the team that you manage in your organisation.": 493,
    "Who needs to improve their skills / competencies / behaviours in order to achieve the key strategic objectives?": 494,
    "Please review the following 20 winning behaviours in relation to yourself and rate how important each is to achieving your organisation's strategic objectives.": 495,
    "Please rank the Winning Behaviours you want to prioritize in this training/program.": 497,
    "What key performance indicators do you have relating to learning and development? Please list any, otherwise write N/A.": 498,
    "Please select at most 5 winning behaviours that you think we should prioritise for your MANAGERS in this training/programme.": 499,
    "Please select at most 5 winning behaviours that you think we should prioritise for your NON-MANAGERS in this training/programme.": 500,
    "Are there any winning behaviours we have not identified you believe are important?": 501,
    "To the best of your knowledge, did the participants use the insights resulting from the SCOPE Assessment to further strengthen their capacities or improve business processes, or adjust their positioning or messaging to potential investors?": 502,
    "How would you define strategic planning in the context of Gatsby Africa?": 503,
    "Please list at most 3 expectations from the upcoming strategic planning session.": 504,
    "What specific areas of strategic planning are you hoping to gain knowledge or skills in?": 505,
    "What are your current challenges or concerns regarding strategic planning within Gatsby Africa?": 506,
    "What challenges do you foresee in implementing strategic planning initiatives effectively?": 507,
    "Is there something specific that the organisation should consider doing differently in terms of strategic planning across departments? If yes, please provide details": 508,
    "Please list the top 2-3 strategic priorities that will drive you towards achieving your Growth Goal:": 509,
    "Please indicate your Growth Goal as a percentage eg. Our goal is to grow our revenue by 15%.": 510,
    "I have reflected with my team on the core strategic objectives that will ensure we achieve our Growth Goal:": 511,
    "Which smartphone do you use to access apps and the Internet?": 512,
    "Accessing online resources and tools is a core part of this programme. Do you have access to reliable internet?": 513,
    "On average, how many MSMEs do you support per year?": 518,
    "Do you own a business? If yes, please provide the name of your business. If not, please indicate N/A": 524,
    "Do you or your organisation operate full time?": 525,
    "Please also describe what your main product/Service is": 526,
    "What form of business development support do you provide? (You can select more than 1)": 527,
    "Does your business address any Sustainable Development Goals (SDGs) either in the work you do, or how you execute your strategy?": 529,
    "I am most at ease when the success or failure of my business depends primarily on my own actions.": 530,
    "I generally find/notice and begin to solve problems before anyone else does.": 531,
    "I would rather get on with managing my business now than spend time planning for the future.": 532,
    "I often keep trying even when others believe that the barriers to success are too high.": 533,
    "I expect business conditions will be better in the future than they are in the present.": 534,
    "When faced with setbacks, I can quickly bounce back and continue working towards my business goals.": 535,
    "I tend to be the person who makes sure the job gets done.": 536,
    "I actively seek ways to improve or adapt existing ideas, processes, and products in my business.": 537,
    "I believe that with effort and learning, I can continue to improve as an entrepreneur.": 538,
    "I am comfortable taking calculated risks to grow my business.": 539,
    "Is your business co-owned?": 540,
    "Since engaging with the programme, have you engaged in any new collaborative and strategic partnerships?": 541,
    "How useful did you find the strategic partnerships you were engaged in?": 542,
    "What type of business registration do you hold?": 547,
    "What do you primarily do within the Agriculture industry?": 548
}

# ============= CSS STYLES =============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .warning-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .conflict-card {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .info-card {
        background: #d1ecf1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .data-source-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============= SESSION STATE INITIALIZATION =============
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "page": "home",
        "df_target": None,
        "df_final": None,
        "uid_changes": {},
        "custom_questions": pd.DataFrame(columns=["Customized Question", "Original Question", "Final_UID"]),
        "question_bank": None,
        "question_bank_with_authority": None,
        "survey_template": None,
        "preview_df": None,
        "all_questions": None,
        "dedup_questions": [],
        "dedup_choices": [],
        "pending_survey": None,
        "snowflake_initialized": False,
        "surveymonkey_initialized": False,
        "uid_conflicts_summary": None,
        "primary_matching_reference": None,
        "fetched_survey_ids": [],
        "categorized_questions": None,
        "uid_final_reference": UID_FINAL_REFERENCE,
        "unique_uid_table": None,
        "edited_df": pd.DataFrame(columns=["question_text", "schema_type", "is_choice", "required"])
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ============= CACHED RESOURCES =============
@st.cache_resource
def load_sentence_transformer():
    """Load the sentence transformer model"""
    logger.info(f"Loading SentenceTransformer model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def get_snowflake_engine():
    """Create and cache Snowflake connection engine"""
    try:
        sf = st.secrets["snowflake"]
        logger.info(f"Attempting Snowflake connection: user={sf.user}, account={sf.account}")
        engine = create_engine(
            f"snowflake://{sf.user}:{sf.password}@{sf.account}/{sf.database}/{sf.schema}"
            f"?warehouse={sf.warehouse}&role={sf.role}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT CURRENT_VERSION()"))
        return engine
    except Exception as e:
        logger.error(f"Snowflake engine creation failed: {e}")
        if "250001" in str(e):
            st.warning(
                "🔒 Snowflake connection failed: User account is locked. "
                "UID matching is disabled, but you can use SurveyMonkey features."
            )
        raise

@st.cache_data
def get_tfidf_vectors(df_reference):
    """Create and cache TF-IDF vectors"""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(df_reference["norm_text"])
    return vectorizer, vectors

@st.cache_data
def load_uid_final_reference_as_df():
    """Load UID Final Reference as a cached DataFrame"""
    uid_final_df = pd.DataFrame([
        {"question_text": question, "uid_final": uid_final}
        for question, uid_final in UID_FINAL_REFERENCE.items()
    ])
    uid_final_df = uid_final_df.sort_values('uid_final').reset_index(drop=True)
    return uid_final_df

# ============= UTILITY FUNCTIONS =============
def enhanced_normalize(text, synonym_map=ENHANCED_SYNONYM_MAP):
    """Enhanced text normalization with synonym mapping"""
    if not isinstance(text, str):
        return ""
    try:
        text = text.lower().strip()
        # Apply synonym mapping
        for phrase, replacement in synonym_map.items():
            text = text.replace(phrase, replacement)
        
        # Remove punctuation and normalize spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove stop words
        words = text.split()
        words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
        return ' '.join(words)
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        return ""

def score_question_quality(question):
    """Enhanced question quality scoring with English structure preference"""
    try:
        if not isinstance(question, str) or len(question.strip()) < 5:
            return 0
        
        score = 0
        text = question.lower().strip()
        original_text = question.strip()
        
        # Length scoring (optimal range 10-100 characters)
        length = len(question)
        if 10 <= length <= 100:
            score += 25
        elif 5 <= length <= 150:
            score += 15
        elif length < 5:
            score -= 25
        
        # Question format scoring
        if original_text.endswith('?'):
            score += 30
        
        # English question words at the beginning
        question_words = ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should']
        first_three_words = text.split()[:3]
        if any(word in first_three_words for word in question_words):
            score += 25
        
        # Proper capitalization
        if question and question[0].isupper():
            score += 15
        
        # Grammar and structure bonuses
        if ' is ' in text or ' are ' in text:
            score += 10
        
        # Complete sentence structure
        word_count = len(question.split())
        if 3 <= word_count <= 15:
            score += 20
        elif word_count < 3:
            score -= 20
        
        # Avoid common artifacts
        bad_patterns = [
            'click here', 'please select', '...', 'n/a', 'other', 
            'select one', 'choose all', 'privacy policy', 'thank you',
            'contact us', 'submit', 'continue', '<div', '<span', 'html'
        ]
        if any(pattern in text for pattern in bad_patterns):
            score -= 30
        
        # Avoid HTML content
        if '<' in question and '>' in question:
            score -= 40
        
        # Bonus for well-formed questions
        if original_text.endswith('?') and any(word in first_three_words for word in question_words):
            score += 15
        
        return max(0, score)
        
    except Exception as e:
        logger.error(f"Error scoring question quality: {e}")
        return 0

def get_best_question_for_uid(variants, occurrence_counts=None):
    """Select the best quality question from variants"""
    try:
        if not variants:
            return None
        
        valid_variants = [v for v in variants if isinstance(v, str) and len(v.strip()) > 3]
        if not valid_variants:
            return None
        
        # If occurrence counts provided, prioritize by highest authority count first
        if occurrence_counts and isinstance(occurrence_counts, dict):
            variant_scores = []
            for variant in valid_variants:
                count = occurrence_counts.get(variant, 0)
                quality = score_question_quality(variant)
                variant_scores.append((variant, count, quality))
            
            variant_scores.sort(key=lambda x: (-x[1], -x[2], len(x[0])))
            return variant_scores[0][0]
        
        # Fallback to quality-based selection
        scored_variants = [(v, score_question_quality(v)) for v in valid_variants]
        
        def sort_key(item):
            question, score = item
            has_question_mark = question.strip().endswith('?')
            has_question_word = any(question.lower().strip().startswith(word) for word in ['what', 'how', 'when', 'where', 'why', 'which', 'do', 'does', 'did', 'are', 'is', 'was', 'were', 'can', 'will', 'would', 'should'])
            proper_structure_bonus = 1000 if (has_question_mark and has_question_word) else 0
            
            return (-score - proper_structure_bonus, len(question))
        
        scored_variants.sort(key=sort_key)
        return scored_variants[0][0]
    except Exception as e:
        logger.error(f"Error selecting best question: {e}")
        return None

def classify_question(text, heading_references=HEADING_REFERENCES):
    """Classify question as Heading or Main Question"""
    try:
        # Length-based heuristic
        if len(text.split()) > HEADING_LENGTH_THRESHOLD:
            return "Heading"
        
        # Check against heading references
        for ref in heading_references:
            if text.strip() in ref or ref in text.strip():
                return "Heading"
        
        # TF-IDF similarity
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            all_texts = heading_references + [text]
            tfidf_vectors = vectorizer.fit_transform([enhanced_normalize(t) for t in all_texts])
            similarity_scores = cosine_similarity(tfidf_vectors[-1], tfidf_vectors[:-1])
            max_tfidf_score = np.max(similarity_scores)
            
            # Semantic similarity
            model = load_sentence_transformer()
            emb_text = model.encode([text], convert_to_tensor=True)
            emb_refs = model.encode(heading_references, convert_to_tensor=True)
            semantic_scores = util.cos_sim(emb_text, emb_refs)[0]
            max_semantic_score = np.max(semantic_scores.cpu().numpy())
            
            # Combine criteria
            if max_tfidf_score >= HEADING_TFIDF_THRESHOLD or max_semantic_score >= HEADING_SEMANTIC_THRESHOLD:
                return "Heading"
        except Exception as e:
            logger.error(f"Question classification failed: {e}")
        
        return "Main Question/Multiple Choice"
    except Exception as e:
        logger.error(f"Error in classify_question: {e}")
        return "Main Question/Multiple Choice"

def calculate_matched_percentage(df_final):
    """Calculate percentage of matched questions"""
    if df_final is None or df_final.empty:
        return 0.0
    df_main = df_final[df_final["is_choice"] == False].copy()
    privacy_filter = ~df_main["question_text"].str.contains("Our Privacy Policy", case=False, na=False)
    html_pattern = r"<div.*text-align:\s*center.*<span.*font-size:\s*12pt.*<em>If you have any questions, please contact your AMI Learner Success Manager.*</em>.*</span>.*</div>"
    html_filter = ~df_main["question_text"].str.contains(html_pattern, case=False, na=False, regex=True)
    eligible_questions = df_main[privacy_filter & html_filter]
    if eligible_questions.empty:
        return 0.0
    matched_questions = eligible_questions[eligible_questions["Final_UID"].notna()]
    return round((len(matched_questions) / len(eligible_questions)) * 100, 2)

def match_question_to_uid_final(question_text):
    """Match a question to UID Final using the reference mapping"""
    uid_final_ref = st.session_state.get('uid_final_reference', {})
    
    # Direct match first
    if question_text in uid_final_ref:
        return uid_final_ref[question_text]
    
    # Try fuzzy matching for minor variations
    question_normalized = enhanced_normalize(question_text)
    
    for ref_question, uid_final in uid_final_ref.items():
        ref_normalized = enhanced_normalize(ref_question)
        
        # Simple similarity check
        if question_normalized == ref_normalized:
            return uid_final
        
        # Check if one contains the other (for partial matches)
        if len(question_normalized) > 10 and len(ref_normalized) > 10:
            if question_normalized in ref_normalized or ref_normalized in question_normalized:
                return uid_final
    
    return None

def categorize_survey_by_ami_structure(title):
    """Categorize survey based on AMI structure: Survey Stage, Respondent Type, Programme"""
    if not isinstance(title, str):
        return {"Survey Stage": "Other", "Respondent Type": "Participant", "Programme": "Custom"}
    
    title_lower = title.lower().strip()
    
    # Determine Survey Stage
    survey_stage = "Other"
    for stage, keywords in SURVEY_STAGES.items():
        if any(keyword.lower() in title_lower for keyword in keywords):
            survey_stage = stage
            break
    
    # Determine Respondent Type
    respondent_type = "Participant"
    for resp_type, keywords in RESPONDENT_TYPES.items():
        if any(keyword.lower() in title_lower for keyword in keywords):
            respondent_type = resp_type
            break
    
    # Determine Programme
    programme = "Custom"
    for prog, keywords in PROGRAMMES.items():
        if any(keyword.lower() in title_lower for keyword in keywords):
            programme = prog
            break
    
    return {
        "Survey Stage": survey_stage,
        "Respondent Type": respondent_type,
        "Programme": programme
    }

# ============= IDENTITY DETECTION FUNCTIONS =============
def contains_identity_info(text):
    """Check if question/choice text contains identity information"""
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower().strip()
    
    # Check for direct matches
    for identity_type in IDENTITY_TYPES:
        if identity_type in text_lower:
            return True
    
    # Additional patterns for identity detection
    identity_patterns = [
        r'\b(name|surname|firstname|lastname)\b',
        r'\b(email|e-mail|mail)\b',
        r'\b(company|organization|organisation)\b',
        r'\b(phone|mobile|telephone|contact)\b',
        r'\b(address|location)\b',
        r'\b(age|gender|sex)\b',
        r'\b(title|position|role|job)\b',
        r'\b(country|region|city|department)\b',
        r'\b(id|identification|passport|pin)\b',
        r'\b(student number|uct)\b',
        r'\b(date of birth|dob|birth)\b',
        r'\b(marital status|married|single)\b',
        r'\b(education|qualification|degree)\b',
        r'\b(english proficiency|language)\b'
    ]
    
    for pattern in identity_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def determine_identity_type(text):
    """Determine the specific identity type from question text"""
    if not isinstance(text, str):
        return 'Unknown'
    
    text_lower = text.lower().strip()
    
     # Priority order for identity type detection
    if any(name in text_lower for name in ['first name', 'firstname']):
        return 'First Name'
    elif any(name in text_lower for name in ['last name', 'lastname', 'surname']):
        return 'Last Name'
    elif any(name in text_lower for name in ['full name']) or ('name' in text_lower and 'first' not in text_lower and 'last' not in text_lower and 'company' not in text_lower):
        return 'Full Name'
    elif any(email in text_lower for email in ['email', 'e-mail', 'mail']):
        return 'E-Mail'
    elif any(company in text_lower for company in ['company', 'organization', 'organisation']):
        return 'Company'
    elif any(phone in text_lower for phone in ['phone', 'mobile', 'telephone']):
        return 'Phone Number'
    elif 'gender' in text_lower or 'sex' in text_lower:
        return 'Gender'
    elif 'age' in text_lower:
        return 'Age'
    elif any(title in text_lower for title in ['title', 'position', 'role', 'job']):
        return 'Title/Role'
    elif 'country' in text_lower:
        return 'Country'
    elif 'region' in text_lower:
        return 'Region'
    elif 'city' in text_lower:
        return 'City'
    elif 'department' in text_lower:
        return 'Department'
    elif any(loc in text_lower for loc in ['location', 'address']):
        return 'Location'
    elif any(id_type in text_lower for id_type in ['id number', 'identification']):
        return 'ID Number'
    elif 'passport' in text_lower or 'pin' in text_lower:
        return 'PIN/ Passport'
    elif 'student number' in text_lower:
        return 'Student Number'
    elif 'uct' in text_lower:
        return 'UCT'
    elif any(dob in text_lower for dob in ['date of birth', 'dob', 'birth']):
        return 'Date of Birth'
    elif 'marital' in text_lower:
        return 'Marital Status'
    elif any(edu in text_lower for edu in ['education', 'qualification', 'degree']):
        return 'Education level'
    elif 'english proficiency' in text_lower or ('language' in text_lower and 'proficiency' in text_lower):
        return 'English Proficiency'
    else:
        return 'Other'

def clean_question_text(text):
    """Clean question text by removing year specifications and extracting core question"""
    if not isinstance(text, str):
        return text
    
    # Remove year specifications like (i.e. 1 Jan. 2024 - 31 Dec. 2024)
    text = re.sub(r'\(i\.e\.\s*\d{1,2}\s+\w+\.?\s+\d{4}\s*-\s*\d{1,2}\s+\w+\.?\s+\d{4}\)', '', text)
    
    # Remove other date patterns
    text = re.sub(r'\(\d{1,2}\s+\w+\.?\s+\d{4}\s*-\s*\d{1,2}\s+\w+\.?\s+\d{4}\)', '', text)
    
    # For mobile number patterns, extract the main question part
    if 'Your mobile number' in text and '<br>' in text:
        parts = text.split('<br>')
        if parts:
            return parts[0].strip()
    
    # For questions with HTML formatting, extract the main question
    if '<br>' in text:
        parts = text.split('<br>')
        for part in parts:
            clean_part = re.sub(r'<[^>]+>', '', part).strip()
            if len(clean_part) > 10 and not clean_part.startswith('Country area'):
                return clean_part
    
    # Clean up extra spaces and HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_question_for_grouping(text):
    """Normalize question text for grouping similar questions"""
    if not isinstance(text, str):
        return text
    
    cleaned = clean_question_text(text)
    normalized = cleaned.lower().strip()
    normalized = re.sub(r'\s*-\s*', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

# ============= CACHE MANAGEMENT =============
def load_cached_survey_data():
    """Load cached survey data if available and recent"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            cache_time = cache.get("timestamp", 0)
            if time.time() - cache_time < 24 * 3600:
                return (
                    pd.DataFrame(cache.get("all_questions", [])),
                    cache.get("dedup_questions", []),
                    cache.get("dedup_choices", [])
                )
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    return None, [], []

def save_cached_survey_data(all_questions, dedup_questions, dedup_choices):
    """Save survey data to cache"""
    cache = {
        "timestamp": time.time(),
        "all_questions": all_questions.to_dict(orient="records") if not all_questions.empty else [],
        "dedup_questions": dedup_questions,
        "dedup_choices": dedup_choices
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# ============= SURVEYMONKEY API FUNCTIONS =============
def get_surveymonkey_token():
    """Get SurveyMonkey API token from secrets with improved error handling"""
    try:
        # Check if secrets exist
        if "surveymonkey" not in st.secrets:
            logger.error("SurveyMonkey secrets not found in st.secrets")
            return None
        
        # Get the token
        token = st.secrets["surveymonkey"]["access_token"]
        
        # Validate token format (SurveyMonkey tokens are typically long strings)
        if not token or len(token) < 10:
            logger.error("SurveyMonkey token appears to be invalid or empty")
            return None
            
        logger.info("SurveyMonkey token retrieved successfully")
        return token
        
    except KeyError as e:
        logger.error(f"SurveyMonkey token key not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to get SurveyMonkey token: {e}")
        return None

def check_surveymonkey_connection():
    """Check SurveyMonkey API connection status with detailed error reporting"""
    try:
        token = get_surveymonkey_token()
        if not token:
            return False, "No access token available - check secrets configuration"
        
        # Test API call with better error handling
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            "https://api.surveymonkey.com/v3/users/me", 
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_data = response.json()
            username = user_data.get("username", "Unknown")
            return True, f"Connected successfully as {username}"
        elif response.status_code == 401:
            return False, "Authentication failed - invalid token"
        elif response.status_code == 403:
            return False, "Access forbidden - check token permissions"
        elif response.status_code == 429:
            return False, "Rate limit exceeded - try again later"
        else:
            return False, f"API error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return False, "Connection timeout - check internet connection"
    except requests.exceptions.ConnectionError:
        return False, "Connection error - unable to reach SurveyMonkey API"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

@st.cache_data
def get_surveys_cached(token):
    """Get all surveys from SurveyMonkey API"""
    url = "https://api.surveymonkey.com/v3/surveys"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("data", [])

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(requests.HTTPError)
)
def get_survey_details_with_retry(survey_id, token):
    """Get detailed survey information with retry logic"""
    url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 429:
        raise requests.HTTPError("429 Too Many Requests")
    response.raise_for_status()
    return response.json()

def extract_questions(survey_json):
    """Extract questions from SurveyMonkey survey JSON"""
    questions = []
    global_position = 0
    
    for page in survey_json.get("pages", []):
        for question in page.get("questions", []):
            q_text = question.get("headings", [{}])[0].get("heading", "")
            q_id = question.get("id", None)
            family = question.get("family", None)
            
            # Determine schema type
            if family == "single_choice":
                schema_type = "Single Choice"
            elif family == "multiple_choice":
                schema_type = "Multiple Choice"
            elif family == "open_ended":
                schema_type = "Open-Ended"
            elif family == "matrix":
                schema_type = "Matrix"
            else:
                choices = question.get("answers", {}).get("choices", [])
                schema_type = "Multiple Choice" if choices else "Open-Ended"
                if choices and ("select one" in q_text.lower() or len(choices) <= 2):
                    schema_type = "Single Choice"
            
            question_category = classify_question(q_text)
            
            if q_text:
                global_position += 1
                questions.append({
                    "question_text": q_text,
                    "position": global_position,
                    "is_choice": False,
                    "parent_question": None,
                    "question_uid": q_id,
                    "schema_type": schema_type,
                    "mandatory": False,
                    "mandatory_editable": True,
                    "survey_id": survey_json.get("id", ""),
                    "survey_title": survey_json.get("title", ""),
                    "question_category": question_category
                })
                
                # Add choices
                choices = question.get("answers", {}).get("choices", [])
                for choice in choices:
                    choice_text = choice.get("text", "")
                    if choice_text:
                        questions.append({
                            "question_text": f"{q_text} - {choice_text}",
                            "position": global_position,
                            "is_choice": True,
                            "parent_question": q_text,
                            "question_uid": q_id,
                            "schema_type": schema_type,
                            "mandatory": False,
                            "mandatory_editable": False,
                            "survey_id": survey_json.get("id", ""),
                            "survey_title": survey_json.get("title", ""),
                            "question_category": "Main Question/Multiple Choice"
                        })
    return questions

# ============= SNOWFLAKE DATABASE FUNCTIONS =============
def check_snowflake_connection():
    """Check Snowflake database connection status"""
    try:
        engine = get_snowflake_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT CURRENT_VERSION()"))
            version = result.fetchone()[0]
            return True, f"Connected to Snowflake version {version}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def run_snowflake_reference_query(limit=10000, offset=0):
    """Run basic Snowflake reference query for question bank"""
    query = """
        SELECT HEADING_0, MAX(UID) AS UID
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE HEADING_0 IS NOT NULL AND UID IS NOT NULL
        GROUP BY HEADING_0
        LIMIT :limit OFFSET :offset
    """
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn, params={"limit": limit, "offset": offset})
        # Ensure consistent column naming
        result.columns = result.columns.str.lower()
        result = result.rename(columns={'heading_0': 'HEADING_0', 'uid': 'UID'})
        return result
    except Exception as e:
        logger.error(f"Snowflake reference query failed: {e}")
        if "250001" in str(e):
            st.warning("Snowflake connection failed: User account is locked. UID matching is disabled.")
        raise

@st.cache_data(ttl=600)
def get_question_bank_with_authority_count():
    """Fetch question bank with authority count and UID Final reference"""
    query = """
    SELECT 
        HEADING_0, 
        UID, 
        COUNT(*) as AUTHORITY_COUNT
    FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
    WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL 
    AND TRIM(HEADING_0) != ''
    GROUP BY HEADING_0, UID
    ORDER BY UID, AUTHORITY_COUNT DESC
    """
    
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn)
        
        result.columns = result.columns.str.upper()
        
        # Add UID Final column using the reference mapping
        result['UID_FINAL'] = result['HEADING_0'].apply(match_question_to_uid_final)
        
        logger.info(f"Question bank with authority count and UID Final fetched: {len(result)} records")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch question bank with authority count: {e}")
        # Fallback to simple query
        try:
            simple_query = """
            SELECT HEADING_0, MAX(UID) AS UID, 1 AS AUTHORITY_COUNT
            FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
            WHERE UID IS NOT NULL AND HEADING_0 IS NOT NULL
            GROUP BY HEADING_0
            ORDER BY CAST(UID AS INTEGER) ASC
            """
            with get_snowflake_engine().connect() as conn:
                result = pd.read_sql(text(simple_query), conn)
            
            result.columns = result.columns.str.upper()
            # Add UID Final column
            result['UID_FINAL'] = result['HEADING_0'].apply(match_question_to_uid_final)
            
            logger.info(f"Question bank fallback query successful: {len(result)} records")
            return result
            
        except Exception as e2:
            logger.error(f"Both enhanced and fallback queries failed: {e2}")
            return pd.DataFrame()

@st.cache_data(ttl=600)
def get_configured_surveys_from_snowflake():
    """Get distinct survey IDs that are configured in Snowflake"""
    query = """
        SELECT DISTINCT SURVEY_ID
        FROM AMI_DBT.DBT_SURVEY_MONKEY.SURVEY_DETAILS_RESPONSES_COMBINED_LIVE
        WHERE HEADING_0 IS NOT NULL AND UID IS NOT NULL
        GROUP BY SURVEY_ID 
        ORDER BY SURVEY_ID
    """
    try:
        with get_snowflake_engine().connect() as conn:
            result = pd.read_sql(text(query), conn)
        return result['SURVEY_ID'].tolist()
    except Exception as e:
        logger.error(f"Failed to get configured surveys: {e}")
        return []

def count_configured_surveys_from_surveymonkey(surveys):
    """Count how many SurveyMonkey surveys are configured in Snowflake"""
    try:
        configured_surveys = get_configured_surveys_from_snowflake()
        surveymonkey_survey_ids = [str(survey['id']) for survey in surveys]
        configured_count = len([sid for sid in surveymonkey_survey_ids if sid in configured_surveys])
        return configured_count
    except Exception as e:
        logger.error(f"Failed to count configured surveys: {e}")
        return 0

# ============= CONNECTION VALIDATION FUNCTIONS =============
def validate_secrets_configuration():
    """Validate that all required secrets are properly configured"""
    missing_secrets = []
    invalid_secrets = []
    
    # Check SurveyMonkey secrets
    try:
        if "surveymonkey" not in st.secrets:
            missing_secrets.append("surveymonkey")
        else:
            sm_secrets = st.secrets["surveymonkey"]
            if "access_token" not in sm_secrets:
                missing_secrets.append("surveymonkey.access_token")
            elif not sm_secrets["access_token"] or len(sm_secrets["access_token"]) < 10:
                invalid_secrets.append("surveymonkey.access_token (too short or empty)")
    except Exception as e:
        missing_secrets.append(f"surveymonkey (error: {e})")
    
    # Check Snowflake secrets
    try:
        if "snowflake" not in st.secrets:
            missing_secrets.append("snowflake")
        else:
            sf_secrets = st.secrets["snowflake"]
            required_sf_keys = ["user", "password", "account", "database", "schema", "warehouse", "role"]
            for key in required_sf_keys:
                if key not in sf_secrets:
                    missing_secrets.append(f"snowflake.{key}")
                elif not sf_secrets[key]:
                    invalid_secrets.append(f"snowflake.{key} (empty)")
    except Exception as e:
        missing_secrets.append(f"snowflake (error: {e})")
    
    return missing_secrets, invalid_secrets

def initialize_connections_with_better_errors():
    """Initialize connections with detailed error reporting"""
    
    # Validate secrets first
    missing_secrets, invalid_secrets = validate_secrets_configuration()
    
    if missing_secrets or invalid_secrets:
        st.markdown('<div class="conflict-card">', unsafe_allow_html=True)
        st.markdown("### ❌ Configuration Issues Detected")
        
        if missing_secrets:
            st.markdown("**Missing Secrets:**")
            for secret in missing_secrets:
                st.markdown(f"• `{secret}`")
        
        if invalid_secrets:
            st.markdown("**Invalid Secrets:**")
            for secret in invalid_secrets:
                st.markdown(f"• `{secret}`")
        
        st.markdown("**How to fix:**")
        st.markdown("1. Go to your Streamlit app settings")
        st.markdown("2. Navigate to the 'Secrets' section")
        st.markdown("3. Add the missing/invalid secrets in TOML format:")
        
        st.code("""
[surveymonkey]
access_token = "your_surveymonkey_token_here"

[snowflake]
user = "your_username"
password = "your_password"
account = "your_account"
database = "your_database"
schema = "your_schema"
warehouse = "your_warehouse"
role = "your_role"
        """, language="toml")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return False
    
    return True

def safe_initialize_app():
    """Safely initialize the app with better error handling"""
    
    # First validate configuration
    if not initialize_connections_with_better_errors():
        st.stop()
    
    # Initialize connections
    try:
        # Test SurveyMonkey connection
        sm_status, sm_msg = check_surveymonkey_connection()
        
        if not sm_status:
            st.markdown('<div class="warning-card">', unsafe_allow_html=True)
            st.markdown(f"⚠️ **SurveyMonkey Connection Issue:** {sm_msg}")
            st.markdown("</div>", unsafe_allow_html=True)
            surveys = []
            token = None
        else:
            token = get_surveymonkey_token()
            try:
                surveys = get_surveys_cached(token) if token else []
            except Exception as e:
                st.warning(f"Failed to load surveys: {e}")
                surveys = []
        
        # Test Snowflake connection
        sf_status, sf_msg = check_snowflake_connection()
        
        if not sf_status:
            st.markdown('<div class="warning-card">', unsafe_allow_html=True)
            st.markdown(f"⚠️ **Snowflake Connection Issue:** {sf_msg}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        return surveys, token, sm_status, sm_msg, sf_status, sf_msg
        
    except Exception as e:
        st.error(f"❌ Application initialization failed: {e}")
        return [], None, False, str(e), False, "Not tested"

# ============= DATA PROCESSING FUNCTIONS =============
def get_unique_questions_by_category():
    """Extract unique questions per category from ALL cached survey data"""
    # First try to load from cache
    if st.session_state.all_questions is None or st.session_state.all_questions.empty:
        cached_questions, cached_dedup_questions, cached_dedup_choices = load_cached_survey_data()
        if cached_questions is not None and not cached_questions.empty:
            st.session_state.all_questions = cached_questions
            st.session_state.dedup_questions = cached_dedup_questions
            st.session_state.dedup_choices = cached_dedup_choices
            st.session_state.fetched_survey_ids = cached_questions["survey_id"].unique().tolist()
    
    # If still no data, fetch ALL surveys from SurveyMonkey directly
    if st.session_state.all_questions is None or st.session_state.all_questions.empty:
        token = get_surveymonkey_token()
        if token:
            try:
                with st.spinner("🔄 Fetching ALL surveys from SurveyMonkey for categorization..."):
                    surveys = get_surveys_cached(token)
                    combined_questions = []
                    
                    # Fetch all surveys (limit to reasonable number for performance)
                    surveys_to_process = surveys[:50]  # Limit to first 50 surveys
                    
                    progress_bar = st.progress(0)
                    for i, survey in enumerate(surveys_to_process):
                        survey_id = survey['id']
                        try:
                            survey_json = get_survey_details_with_retry(survey_id, token)
                            questions = extract_questions(survey_json)
                            combined_questions.extend(questions)
                            time.sleep(REQUEST_DELAY)
                            progress_bar.progress((i + 1) / len(surveys_to_process))
                        except Exception as e:
                            logger.error(f"Failed to fetch survey {survey_id}: {e}")
                            continue
                    
                    progress_bar.empty()
                    
                    if combined_questions:
                        st.session_state.all_questions = pd.DataFrame(combined_questions)
                        st.session_state.dedup_questions = sorted(st.session_state.all_questions[
                            st.session_state.all_questions["is_choice"] == False
                        ]["question_text"].unique().tolist())
                        st.session_state.dedup_choices = sorted(st.session_state.all_questions[
                            st.session_state.all_questions["is_choice"] == True
                        ]["question_text"].apply(lambda x: x.split(" - ", 1)[1] if " - " in x else x).unique().tolist())
                        
                        # Save to cache
                        save_cached_survey_data(
                            st.session_state.all_questions,
                            st.session_state.dedup_questions,
                            st.session_state.dedup_choices
                        )
                        
                        st.success(f"✅ Fetched {len(combined_questions)} questions from {len(surveys_to_process)} surveys")
                    
            except Exception as e:
                logger.error(f"Failed to fetch surveys for categorization: {e}")
                st.error(f"❌ Failed to fetch surveys: {str(e)}")
                return pd.DataFrame()
    
    all_questions_df = st.session_state.all_questions
    
    if all_questions_df is None or all_questions_df.empty:
        return pd.DataFrame()
    
    try:
        # Add AMI structure categorization
        categorization_data = all_questions_df['survey_title'].apply(categorize_survey_by_ami_structure)
        categorization_df = pd.DataFrame(categorization_data.tolist())
        all_questions_df = pd.concat([all_questions_df, categorization_df], axis=1)
        
        # Clean and normalize question text
        all_questions_df['cleaned_question_text'] = all_questions_df['question_text'].apply(clean_question_text)
        all_questions_df['normalized_question'] = all_questions_df['cleaned_question_text'].apply(normalize_question_for_grouping)
        
        # Group by AMI structure and get unique questions
        category_questions = []
        
        # Get unique combinations of Survey Stage, Respondent Type, and Programme
        unique_combinations = all_questions_df[['Survey Stage', 'Respondent Type', 'Programme']].drop_duplicates()
        
        for _, combo in unique_combinations.iterrows():
            survey_stage = combo['Survey Stage']
            respondent_type = combo['Respondent Type']
            programme = combo['Programme']
            
            category_df = all_questions_df[
                (all_questions_df['Survey Stage'] == survey_stage) &
                (all_questions_df['Respondent Type'] == respondent_type) &
                (all_questions_df['Programme'] == programme)
            ]
            
            if not category_df.empty:
                # Get unique main questions (not choices) by normalized text
                main_questions_df = category_df[category_df['is_choice'] == False].copy()
                if not main_questions_df.empty:
                    unique_main_questions = main_questions_df.groupby('normalized_question').first()
                    
                    # Add main questions
                    for norm_question, question_data in unique_main_questions.iterrows():
                        # Count surveys for this normalized question
                        survey_count = len(main_questions_df[
                            main_questions_df['normalized_question'] == norm_question
                        ]['survey_id'].unique())
                        
                        category_questions.append({
                            'Survey Stage': survey_stage,
                            'Respondent Type': respondent_type,
                            'Programme': programme,
                            'question_text': question_data['cleaned_question_text'],
                            'schema_type': question_data.get('schema_type'),
                            'is_choice': False,
                            'parent_question': None,
                            'survey_count': survey_count,
                            'Final_UID': None,
                            'configured_final_UID': None,
                            'Change_UID': None,
                            'required': False
                        })
                
                # Get unique choices by normalized text
                choices_df = category_df[category_df['is_choice'] == True].copy()
                if not choices_df.empty:
                    unique_choices = choices_df.groupby('normalized_question').first()
                    
                    # Add choices
                    for norm_question, choice_data in unique_choices.iterrows():
                        # Count surveys for this normalized choice
                        survey_count = len(choices_df[
                            choices_df['normalized_question'] == norm_question
                        ]['survey_id'].unique())
                        
                        category_questions.append({
                            'Survey Stage': survey_stage,
                            'Respondent Type': respondent_type,
                            'Programme': programme,
                            'question_text': choice_data['cleaned_question_text'],
                            'schema_type': choice_data.get('schema_type'),
                            'is_choice': True,
                            'parent_question': choice_data.get('parent_question'),
                            'survey_count': survey_count,
                            'Final_UID': None,
                            'configured_final_UID': None,
                            'Change_UID': None,
                            'required': False
                        })
        
        return pd.DataFrame(category_questions)
        
    except Exception as e:
        logger.error(f"Error in get_unique_questions_by_category: {e}")
        st.error(f"❌ Error processing categorized questions: {str(e)}")
        return pd.DataFrame()

def create_unique_uid_table(question_bank_with_authority):
    """Create a table with unique questions per UID using UID Final reference or authority ranking"""
    try:
        if question_bank_with_authority.empty:
            return pd.DataFrame()
        
        # Method 1: Use UID Final reference if available
        uid_final_ref = st.session_state.get('uid_final_reference', {})
        
        if uid_final_ref:
            # Create mapping from UID Final reference
            unique_uid_data = []
            processed_uids = set()
            
             # First, add questions from UID Final reference
            for question_text, uid_final in uid_final_ref.items():
                if uid_final not in processed_uids:
                    # Find corresponding record in question bank
                    matching_records = question_bank_with_authority[
                        question_bank_with_authority['HEADING_0'] == question_text
                    ]
                    
                    if not matching_records.empty:
                        best_record = matching_records.iloc[0]
                        unique_uid_data.append({
                            'UID': best_record['UID'],
                            'UID_FINAL': uid_final,
                            'HEADING_0': question_text,
                            'AUTHORITY_COUNT': best_record.get('AUTHORITY_COUNT', 1),
                            'SOURCE': 'UID Final Reference'
                        })
                        processed_uids.add(uid_final)
            
            # Then add remaining UIDs from question bank that don't have UID Final
            remaining_records = question_bank_with_authority[
                ~question_bank_with_authority['UID_FINAL'].isin(processed_uids)
            ]
            
            if not remaining_records.empty:
                # Group by UID and take the one with highest authority count
                uid_groups = remaining_records.groupby('UID')
                
                for uid, group in uid_groups:
                    best_record = group.loc[group['AUTHORITY_COUNT'].idxmax()]
                    unique_uid_data.append({
                        'UID': best_record['UID'],
                        'UID_FINAL': best_record.get('UID_FINAL'),
                        'HEADING_0': best_record['HEADING_0'],
                        'AUTHORITY_COUNT': best_record['AUTHORITY_COUNT'],
                        'SOURCE': 'Authority Count'
                    })
        
        else:
            # Method 2: Fallback to authority count ranking
            uid_groups = question_bank_with_authority.groupby('UID')
            unique_uid_data = []
            
            for uid, group in uid_groups:
                # Select the record with highest authority count
                best_record = group.loc[group['AUTHORITY_COUNT'].idxmax()]
                unique_uid_data.append({
                    'UID': best_record['UID'],
                    'UID_FINAL': best_record.get('UID_FINAL'),
                    'HEADING_0': best_record['HEADING_0'],
                    'AUTHORITY_COUNT': best_record['AUTHORITY_COUNT'],
                    'SOURCE': 'Authority Count'
                })
        
        unique_uid_df = pd.DataFrame(unique_uid_data)
        
        # Sort by UID
        if not unique_uid_df.empty:
            unique_uid_df = unique_uid_df.sort_values('UID').reset_index(drop=True)
        
        return unique_uid_df
        
    except Exception as e:
        logger.error(f"Error creating unique UID table: {e}")
        return pd.DataFrame()

# ============= OPTIMIZED UID MATCHING FUNCTIONS =============
@st.cache_data(ttl=3600)
def precompute_reference_embeddings(question_bank):
    """Pre-compute and cache reference embeddings for semantic matching"""
    try:
        model = load_sentence_transformer()
        reference_texts = question_bank["HEADING_0"].tolist()
        embeddings = model.encode(reference_texts, convert_to_tensor=True, batch_size=32)
        return embeddings
    except Exception as e:
        logger.error(f"Failed to precompute reference embeddings: {e}")
        return None

def run_uid_match_optimized(question_bank, df_target):
    """Optimized UID matching algorithm with pre-computed embeddings and batch processing"""
    try:
        # Prepare question bank with normalized text
        question_bank_norm = question_bank.copy()
        question_bank_norm["norm_text"] = question_bank_norm["HEADING_0"].apply(enhanced_normalize)
        
        # Prepare target questions
        df_target_norm = df_target.copy()
        df_target_norm["norm_text"] = df_target_norm["question_text"].apply(enhanced_normalize)
        
        # Get TF-IDF vectors (cached)
        vectorizer, reference_vectors = get_tfidf_vectors(question_bank_norm)
        
        # Pre-compute semantic embeddings (cached)
        reference_embeddings = precompute_reference_embeddings(question_bank_norm)
        
        # Initialize results
        df_target_norm["Final_UID"] = None
        df_target_norm["Match_Confidence"] = None
        df_target_norm["Final_Match_Type"] = None
        
        # Load sentence transformer for batch processing
        model = load_sentence_transformer()
        
        # Process all questions at once for TF-IDF
        target_vectors = vectorizer.transform(df_target_norm["norm_text"])
        tfidf_similarities = cosine_similarity(target_vectors, reference_vectors)
        
        # Batch process semantic matching for questions that need it
        semantic_needed_indices = []
        semantic_questions = []
        
        # First pass: TF-IDF matching
        for i, (idx, row) in enumerate(df_target_norm.iterrows()):
            tfidf_scores = tfidf_similarities[i]
            max_tfidf_idx = np.argmax(tfidf_scores)
            max_tfidf_score = tfidf_scores[max_tfidf_idx]
            
            if max_tfidf_score >= TFIDF_HIGH_CONFIDENCE:
                matched_uid = question_bank_norm.iloc[max_tfidf_idx]["UID"]
                df_target_norm.at[idx, "Final_UID"] = matched_uid
                df_target_norm.at[idx, "Match_Confidence"] = "✅ High"
                df_target_norm.at[idx, "Final_Match_Type"] = "✅ High"
            elif max_tfidf_score >= TFIDF_LOW_CONFIDENCE:
                matched_uid = question_bank_norm.iloc[max_tfidf_idx]["UID"]
                df_target_norm.at[idx, "Final_UID"] = matched_uid
                df_target_norm.at[idx, "Match_Confidence"] = "⚠️ Low"
                df_target_norm.at[idx, "Final_Match_Type"] = "⚠️ Low"
            else:
                # Mark for semantic matching
                semantic_needed_indices.append((i, idx))
                semantic_questions.append(row["question_text"])
        
        # Batch semantic matching for remaining questions
        if semantic_questions and reference_embeddings is not None:
            try:
                # Encode all unmatched questions at once
                target_embeddings = model.encode(semantic_questions, convert_to_tensor=True, batch_size=32)
                
                # Calculate similarities in batch
                semantic_similarities = util.cos_sim(target_embeddings, reference_embeddings)
                
                # Process semantic results
                for batch_idx, (original_idx, df_idx) in enumerate(semantic_needed_indices):
                    semantic_scores = semantic_similarities[batch_idx]
                    max_semantic_score = max(semantic_scores).item()
                    
                    if max_semantic_score >= SEMANTIC_THRESHOLD:
                        max_semantic_idx = semantic_scores.argmax().item()
                        matched_uid = question_bank_norm.iloc[max_semantic_idx]["UID"]
                        df_target_norm.at[df_idx, "Final_UID"] = matched_uid
                        df_target_norm.at[df_idx, "Match_Confidence"] = "🧠 Semantic"
                        df_target_norm.at[df_idx, "Final_Match_Type"] = "🧠 Semantic"
                    else:
                        df_target_norm.at[df_idx, "Final_UID"] = None
                        df_target_norm.at[df_idx, "Match_Confidence"] = "❌ No match"
                        df_target_norm.at[df_idx, "Final_Match_Type"] = "❌ No match"
            except Exception as e:
                logger.error(f"Batch semantic matching failed: {e}")
                # Fallback to no match for remaining questions
                for _, df_idx in semantic_needed_indices:
                    df_target_norm.at[df_idx, "Final_UID"] = None
                    df_target_norm.at[df_idx, "Match_Confidence"] = "❌ No match"
                    df_target_norm.at[df_idx, "Final_Match_Type"] = "❌ No match"
        else:
            # No semantic matching possible, mark remaining as no match
            for _, df_idx in semantic_needed_indices:
                df_target_norm.at[df_idx, "Final_UID"] = None
                df_target_norm.at[df_idx, "Match_Confidence"] = "❌ No match"
                df_target_norm.at[df_idx, "Final_Match_Type"] = "❌ No match"
        
        # Remove normalization column before returning
        df_target_norm = df_target_norm.drop(columns=["norm_text"])
        
        return df_target_norm
        
    except Exception as e:
        logger.error(f"UID matching failed: {e}")
        # Return original dataframe with empty UID columns
        df_target["Final_UID"] = None
        df_target["Match_Confidence"] = "❌ Error"
        df_target["Final_Match_Type"] = "❌ Error"
        return df_target

# Update the original function to use the optimized version
def run_uid_match(question_bank, df_target):
    """Run UID matching algorithm between question bank and target questions (optimized)"""
    return run_uid_match_optimized(question_bank, df_target)

# ============= EXPORT FUNCTIONS =============
def prepare_export_data(df_final):
    """Prepare export data split into identity and non-identity tables"""
    try:
        if df_final is None or df_final.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Filter for main questions only (not choices)
        main_questions = df_final[df_final["is_choice"] == False].copy()
        
        if main_questions.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Add identity classification
        main_questions['is_identity'] = main_questions['question_text'].apply(contains_identity_info)
        main_questions['identity_type'] = main_questions['question_text'].apply(determine_identity_type)
        
        # Split into identity and non-identity questions
        identity_questions = main_questions[main_questions['is_identity'] == True].copy()
        non_identity_questions = main_questions[main_questions['is_identity'] == False].copy()
        
        # Prepare non-identity export (Table 1)
        export_df_non_identity = pd.DataFrame()
        if not non_identity_questions.empty:
            export_df_non_identity = non_identity_questions[[
                'question_uid', 'question_text', 'schema_type', 'Final_UID', 'required'
            ]].copy()
            export_df_non_identity.columns = ['question_id', 'question_text', 'question_type', 'UID', 'required']
        
        # Prepare identity export (Table 2)
        export_df_identity = pd.DataFrame()
        if not identity_questions.empty:
            export_df_identity = identity_questions[[
                'question_uid', 'question_text', 'schema_type', 'identity_type', 'Final_UID', 'required'
            ]].copy()
            export_df_identity.columns = ['question_id', 'question_text', 'question_type', 'identity_type', 'UID', 'required']
        
        return export_df_non_identity, export_df_identity
        
    except Exception as e:
        logger.error(f"Failed to prepare export data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def upload_to_snowflake_tables(export_df_non_identity, export_df_identity):
    """Upload both export tables to Snowflake"""
    try:
        engine = get_snowflake_engine()
        
        with st.spinner("🚀 Uploading tables to Snowflake..."):
            # Upload non-identity questions
            if not export_df_non_identity.empty:
                table_name_1 = f"uid_matcher_non_identity_{uuid4().hex[:8]}"
                export_df_non_identity.to_sql(
                    table_name_1, 
                    engine, 
                    if_exists='replace', 
                    index=False,
                    method='multi'
                )
                st.success(f"✅ Non-identity questions uploaded to: {table_name_1}")
            
            # Upload identity questions
            if not export_df_identity.empty:
                table_name_2 = f"uid_matcher_identity_{uuid4().hex[:8]}"
                export_df_identity.to_sql(
                    table_name_2, 
                    engine, 
                    if_exists='replace', 
                    index=False,
                    method='multi'
                )
                st.success(f"✅ Identity questions uploaded to: {table_name_2}")
            
            st.success("🎉 Both tables uploaded successfully to Snowflake!")
        
    except Exception as e:
        logger.error(f"Failed to upload to Snowflake: {e}")
        st.error(f"❌ Failed to upload to Snowflake: {str(e)}")

# ============= MAIN APPLICATION INITIALIZATION =============
# Initialize session state
initialize_session_state()

# Load initial data with improved error handling
surveys, token, sm_status, sm_msg, sf_status, sf_msg = safe_initialize_app()

# Load cached survey data
if st.session_state.all_questions is None:
    cached_questions, cached_dedup_questions, cached_dedup_choices = load_cached_survey_data()
    if cached_questions is not None and not cached_questions.empty:
        st.session_state.all_questions = cached_questions
        st.session_state.dedup_questions = cached_dedup_questions
        st.session_state.dedup_choices = cached_dedup_choices
        st.session_state.fetched_survey_ids = cached_questions["survey_id"].unique().tolist()

# Load question bank
if st.session_state.question_bank is None:
    try:
        st.session_state.question_bank = run_snowflake_reference_query()
    except Exception:
        st.warning("Failed to load question bank. Standardization checks disabled.")
        st.session_state.question_bank = pd.DataFrame(columns=["HEADING_0", "UID"])

# ============= SIDEBAR NAVIGATION =============
with st.sidebar:
    st.markdown("### 🧠 UID Matcher Enhanced")
    st.markdown("Advanced question bank with UID Final reference")
    
    # Connection status
    st.markdown("**🔗 Connection Status**")
    st.write(f"📊 SurveyMonkey: {'✅' if sm_status else '❌'}")
    st.write(f"❄️ Snowflake: {'✅' if sf_status else '❌'}")
    
    # UID Final reference status
    uid_final_count = len(st.session_state.get('uid_final_reference', {}))
    st.write(f"🎯 UID Final Ref: {uid_final_count} items")
    
    # Data source info
    with st.expander("📊 Data Sources"):
        st.markdown("**SurveyMonkey (Source):**")
        st.markdown("• Survey data and questions")
        st.markdown("• question_uid → SurveyMonkey question ID")
        st.markdown("• question_text → SurveyMonkey question/choice text")
        st.markdown("**Snowflake (Reference):**")
        st.markdown("• HEADING_0 → reference questions")
        st.markdown("• UID → target assignment")
        st.markdown("**UID Final Reference:**")
        st.markdown("• HEADING_0 → UID Final mapping")
        st.markdown("• From provided reference file")
    
    # Main navigation
    if st.button("🏠 Home Dashboard", use_container_width=True, key="nav_home"):
        st.session_state.page = "home"
        st.rerun()
    
    st.markdown("---")
    
    # Survey Management
    st.markdown("**📊 Survey Management**")
    if st.button("📋 Survey Selection", use_container_width=True, key="nav_survey_selection"):
        st.session_state.page = "survey_selection"
        st.rerun()
    
    # Survey Categorization
    st.markdown("**📂 AMI Structure**")
    if st.button("📊 AMI Categories", use_container_width=True, key="nav_ami_categories"):
        st.session_state.page = "survey_categorization"
        st.rerun()
    
    if st.button("🔧 UID Matching", use_container_width=True, key="nav_uid_matching"):
        st.session_state.page = "uid_matching"
        st.rerun()
    if st.button("🏗️ Survey Creation", use_container_width=True, key="nav_survey_creation"):
        st.session_state.page = "survey_creation"
        st.rerun()
    
    st.markdown("---")
    
    # Question Bank - SIMPLIFIED
    st.markdown("**📚 Question Bank**")
    if st.button("📖 View Question Bank", use_container_width=True, key="nav_question_bank"):
        st.session_state.page = "question_bank"
        st.rerun()

# ============= MAIN APP HEADER =============
st.markdown('<div class="main-header">🧠 UID Matcher: Enhanced with UID Final Reference</div>', unsafe_allow_html=True)

# Data source clarification
st.markdown('<div class="data-source-info"><strong>📊 Data Flow:</strong> SurveyMonkey surveys → Snowflake reference → UID Final mapping → Enhanced question bank</div>', unsafe_allow_html=True)

# ============= PAGE ROUTING =============

if st.session_state.page == "home":
    st.markdown("## 🏠 Welcome to Enhanced UID Matcher")
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔄 Status", "Active")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if sf_status and st.session_state.question_bank is not None and not st.session_state.question_bank.empty:
            unique_uids = st.session_state.question_bank["UID"].nunique()
            st.metric("❄️ Snowflake UIDs", f"{unique_uids:,}")
        else:
            st.metric("❄️ Snowflake UIDs", "No Connection")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 SM Surveys", len(surveys))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        uid_final_count = len(st.session_state.get('uid_final_reference', {}))
        st.metric("🎯 UID Final Refs", f"{uid_final_count}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # UID Final Reference info
    st.markdown("## 🎯 UID Final Reference")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("**New Feature:** UID Final reference mapping loaded from provided file")
    st.markdown(f"• **{uid_final_count} questions** mapped to UID Final values")
    st.markdown("• Used in Question Bank viewer for enhanced reference")
    st.markdown("• Provides authoritative UID assignments")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Workflow guide
    st.markdown("## 🚀 Recommended Workflow")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Survey Selection")
        st.markdown("Select and analyze surveys:")
        st.markdown("• Browse available surveys")
        st.markdown("• Extract questions with IDs")
        st.markdown("• Review question bank")
        
        if st.button("📋 Start Survey Selection", use_container_width=True, key="workflow_survey_selection"):
            st.session_state.page = "survey_selection"
            st.rerun()
    
    with col2:
        st.markdown("### 2️⃣ AMI Structure")
        st.markdown("Categorize with AMI structure:")
        st.markdown("• Survey Stage classification")
        st.markdown("• Respondent Type grouping")
        st.markdown("• Programme alignment")
        
        if st.button("📊 View AMI Categories", use_container_width=True, key="workflow_ami_categories"):
            st.session_state.page = "survey_categorization"
            st.rerun()
    
    with col3:
        st.markdown("### 3️⃣ Question Bank")
        st.markdown("Enhanced question bank:")
        st.markdown("• Snowflake reference questions")
        st.markdown("• **UID Final reference**")
        st.markdown("• **Unique UID table creation**")
        
        if st.button("📖 View Question Bank", use_container_width=True, key="workflow_question_bank"):
            st.session_state.page = "question_bank"
            st.rerun()
    
    # System status
    st.markdown("---")
    st.markdown("## 🔧 System Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        if sm_status:
            st.markdown('<div class="success-card">✅ SurveyMonkey: Connected</div>', unsafe_allow_html=True)
            st.write(f"Status: {sm_msg}")
        else:
            st.markdown('<div class="warning-card">❌ SurveyMonkey: Connection Issues</div>', unsafe_allow_html=True)
            st.write(f"Error: {sm_msg}")
    
    with status_col2:
        if sf_status:
            st.markdown('<div class="success-card">✅ Snowflake: Connected</div>', unsafe_allow_html=True)
            st.write(f"Status: {sf_msg}")
        else:
            st.markdown('<div class="warning-card">❌ Snowflake: Connection Issues</div>', unsafe_allow_html=True)
            st.write(f"Error: {sf_msg}")

# ============= SURVEY SELECTION PAGE =============
elif st.session_state.page == "survey_selection":
    st.markdown("## 📋 Survey Selection & Question Bank")
    st.markdown('<div class="data-source-info">📊 <strong>Data Source:</strong> SurveyMonkey API - Survey selection and question extraction</div>', unsafe_allow_html=True)
    
    if not surveys:
        st.markdown('<div class="warning-card">⚠️ No surveys available. Check SurveyMonkey connection.</div>', unsafe_allow_html=True)
        st.stop()
    
    # Survey Selection
    st.markdown("### 🔍 Select Surveys")
    survey_options = [f"{s['id']} - {s['title']}" for s in surveys]
    selected_surveys = st.multiselect("Choose surveys to analyze:", survey_options)
    selected_survey_ids = [s.split(" - ")[0] for s in selected_surveys]
    
    # Refresh button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh Survey Data"):
            st.session_state.all_questions = None
            st.session_state.dedup_questions = []
            st.session_state.dedup_choices = []
            st.session_state.fetched_survey_ids = []
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            st.rerun()
    
    # Process selected surveys
    if selected_survey_ids and token:
        combined_questions = []
        
        # Check which surveys need to be fetched
        surveys_to_fetch = [sid for sid in selected_survey_ids 
                           if sid not in st.session_state.fetched_survey_ids]
        
        if surveys_to_fetch:
            progress_bar = st.progress(0)
            for i, survey_id in enumerate(surveys_to_fetch):
                with st.spinner(f"Fetching survey {survey_id}... ({i+1}/{len(surveys_to_fetch)})"):
                    try:
                        survey_json = get_survey_details_with_retry(survey_id, token)
                        questions = extract_questions(survey_json)
                        combined_questions.extend(questions)
                        st.session_state.fetched_survey_ids.append(survey_id)
                        time.sleep(REQUEST_DELAY)
                        progress_bar.progress((i + 1) / len(surveys_to_fetch))
                    except Exception as e:
                        st.error(f"Failed to fetch survey {survey_id}: {e}")
                        continue
            progress_bar.empty()
        
        if combined_questions:
            new_questions = pd.DataFrame(combined_questions)
            if st.session_state.all_questions is None:
                st.session_state.all_questions = new_questions
            else:
                st.session_state.all_questions = pd.concat([st.session_state.all_questions, new_questions], ignore_index=True)
            
            st.session_state.dedup_questions = sorted(st.session_state.all_questions[
                st.session_state.all_questions["is_choice"] == False
            ]["question_text"].unique().tolist())
            st.session_state.dedup_choices = sorted(st.session_state.all_questions[
                st.session_state.all_questions["is_choice"] == True
            ]["question_text"].apply(lambda x: x.split(" - ", 1)[1] if " - " in x else x).unique().tolist())
            
            save_cached_survey_data(
                st.session_state.all_questions,
                st.session_state.dedup_questions,
                st.session_state.dedup_choices
            )

        # Filter data for selected surveys
        if st.session_state.all_questions is not None:
            st.session_state.df_target = st.session_state.all_questions[
                st.session_state.all_questions["survey_id"].isin(selected_survey_ids)
            ].copy()
            
            if st.session_state.df_target.empty:
                st.markdown('<div class="warning-card">⚠️ No questions found for selected surveys.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-card">✅ Questions loaded successfully!</div>', unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Questions", len(st.session_state.df_target))
                with col2:
                    main_questions = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == False])
                    st.metric("❓ Main Questions", main_questions)
                with col3:
                    choices = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == True])
                    st.metric("🔘 Choice Options", choices)
                
                st.markdown("### 📋 Selected Questions Preview")
                show_main_only = st.checkbox("Show main questions only", value=True)
                display_df = st.session_state.df_target[st.session_state.df_target["is_choice"] == False] if show_main_only else st.session_state.df_target
                
                # Show questions with question_uid (question_id)
                display_columns = ["question_uid", "question_text", "schema_type", "is_choice", "survey_title"]
                available_columns = [col for col in display_columns if col in display_df.columns]
                st.dataframe(display_df[available_columns], height=400)
                
                # Next step buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📊 Proceed to AMI Categories", type="primary", use_container_width=True):
                        st.session_state.page = "survey_categorization"
                        st.rerun()
                with col2:
                    if st.button("🔧 Proceed to UID Matching", use_container_width=True):
                        st.session_state.page = "uid_matching"
                        st.rerun()

    # Question Bank Section
    st.markdown("---")
    st.markdown("### 📚 Enhanced Question Bank")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👁️ View Enhanced Question Bank", use_container_width=True):
            st.session_state.page = "question_bank"
            st.rerun()
    
    with col2:
        if st.button("➕ Add to Question Bank", use_container_width=True):
            st.markdown("**Submit new questions:**")
            st.markdown("[📝 Question Submission Form](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")

# ============= QUESTION BANK PAGE =============
elif st.session_state.page == "question_bank":
    st.markdown("## 📚 Enhanced Question Bank Viewer")
    st.markdown('<div class="data-source-info">❄️ <strong>Data Source:</strong> Snowflake + UID Final Reference - Enhanced question bank with authoritative mappings</div>', unsafe_allow_html=True)
    
    # Load UID Final Reference as DataFrame for display
    uid_final_df = load_uid_final_reference_as_df()
    
    # Display UID Final Reference first
    st.markdown("### 🎯 UID Final Reference (Cached)")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown(f"**Loaded {len(uid_final_df)} authoritative question-to-UID Final mappings from cached reference**")
    st.markdown("This is the master reference for UID Final assignments, independent of Snowflake connection.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show UID Final Reference metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total UID Final Records", len(uid_final_df))
    with col2:
        unique_uid_finals = uid_final_df['uid_final'].nunique()
        st.metric("🎯 Unique UID Finals", unique_uid_finals)
    with col3:
        min_uid_final = uid_final_df['uid_final'].min()
        max_uid_final = uid_final_df['uid_final'].max()
        st.metric("📈 UID Final Range", f"{min_uid_final}-{max_uid_final}")
    
    # UID Final Reference search and display
    st.markdown("#### 🔍 Search UID Final Reference")
    search_uid_final = st.text_input("Search questions or UID Final:", key="uid_final_search")
    
    # Filter UID Final reference
    filtered_uid_final = uid_final_df.copy()
    if search_uid_final:
        filtered_uid_final = uid_final_df[
            (uid_final_df['question_text'].str.contains(search_uid_final, case=False, na=False)) |
            (uid_final_df['uid_final'].astype(str).str.contains(search_uid_final, case=False, na=False))
        ]
    
    # Display UID Final Reference table
    st.dataframe(
        filtered_uid_final,
        column_config={
            "uid_final": st.column_config.NumberColumn("UID Final", width="small"),
            "question_text": st.column_config.TextColumn("Question Text", width="large")
        },
        use_container_width=True,
        height=300
    )
    
    # Download UID Final Reference
    csv_uid_final = uid_final_df.to_csv(index=False)
    st.download_button(
        "📥 Download UID Final Reference",
        csv_uid_final,
        f"uid_final_reference_{uuid4()}.csv",
        "text/csv",
        key="download_uid_final_ref"
    )
    
    st.markdown("---")
    
    # Snowflake Question Bank (if available) - LOAD DATA BUT DON'T DISPLAY
    if sf_status:
        try:
            # Load data in background for functionality but don't display the enhanced section
            with st.spinner("📊 Loading question bank data in background..."):
                question_bank_with_authority = get_question_bank_with_authority_count()
            
            if not question_bank_with_authority.empty:
                st.session_state.question_bank_with_authority = question_bank_with_authority
                
                # Create Unique UID Table functionality (keep this)
                st.markdown("### 🎯 Unique UID Table Management")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔧 Create Unique UID Table", type="primary", use_container_width=True):
                        with st.spinner("🔄 Creating unique UID table..."):
                            unique_uid_table = create_unique_uid_table(question_bank_with_authority)
                            st.session_state.unique_uid_table = unique_uid_table
                        
                        if not unique_uid_table.empty:
                            st.markdown('<div class="success-card">✅ Unique UID table created successfully!</div>', unsafe_allow_html=True)
                            
                            # Show metrics for unique table
                            col1_inner, col2_inner, col3_inner = st.columns(3)
                            with col1_inner:
                                st.metric("🆔 Unique UIDs", len(unique_uid_table))
                            with col2_inner:
                                uid_final_refs = unique_uid_table['SOURCE'].value_counts().get('UID Final Reference', 0)
                                st.metric("🎯 From UID Final", uid_final_refs)
                            with col3_inner:
                                authority_refs = unique_uid_table['SOURCE'].value_counts().get('Authority Count', 0)
                                st.metric("📊 From Authority", authority_refs)
                        else:
                            st.error("❌ Failed to create unique UID table")
                
                with col2:
                    if hasattr(st.session_state, 'unique_uid_table') and not st.session_state.unique_uid_table.empty:
                        csv_unique = st.session_state.unique_uid_table.to_csv(index=False)
                        st.download_button(
                            "📥 Download Unique UID Table",
                            csv_unique,
                            f"unique_uid_table_{uuid4().hex[:8]}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                # Display unique UID table if it exists
                if hasattr(st.session_state, 'unique_uid_table') and not st.session_state.unique_uid_table.empty:
                    st.markdown("#### 🎯 Unique UID Table Preview")
                    
                    # Define column config properly
                    unique_table_column_config = {
                        "UID": st.column_config.NumberColumn("UID", width="small"),
                        "UID_FINAL": st.column_config.NumberColumn("UID Final", width="medium"),
                        "HEADING_0": st.column_config.TextColumn("Question Text", width="large"),
                        "AUTHORITY_COUNT": st.column_config.NumberColumn("Authority Count", width="medium"),
                        "SOURCE": st.column_config.TextColumn("Source", width="medium")
                    }
                    
                    st.dataframe(
                        st.session_state.unique_uid_table,
                        column_config=unique_table_column_config,
                        use_container_width=True,
                        height=300
                    )
            else:
                st.warning("⚠️ No question bank data available from Snowflake")
        
        except Exception as e:
            logger.error(f"Failed to load question bank: {e}")
            st.error(f"❌ Failed to load question bank: {str(e)}")
    else:
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.markdown("⚠️ **Snowflake connection not available.** Only UID Final Reference is shown above.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============= AMI SURVEY CATEGORIZATION PAGE =============
elif st.session_state.page == "survey_categorization":
    st.markdown("## 📊 AMI Survey Categorization")
    st.markdown('<div class="data-source-info">📂 <strong>Data Source:</strong> SurveyMonkey questions/choices - AMI structure categorization using cached survey data</div>', unsafe_allow_html=True)
    
    # AMI Structure overview
    st.markdown("### 📂 AMI Survey Structure Overview")
    
    with st.expander("📋 AMI Structure Definitions", expanded=False):
        st.markdown("**Survey Stages:**")
        for stage, keywords in SURVEY_STAGES.items():
            st.markdown(f"• **{stage}:** {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
        
        st.markdown("\n**Respondent Types:**")
        for resp_type, keywords in RESPONDENT_TYPES.items():
            st.markdown(f"• **{resp_type}:** {', '.join(keywords)}")
        
        st.markdown("\n**Programmes:**")
        for programme, keywords in PROGRAMMES.items():
            st.markdown(f"• **{programme}:** {', '.join(keywords[:3])}{'...' if len(keywords) > 3 else ''}")
    
    # Generate categorized questions from ALL cached survey data using AMI structure
    with st.spinner("📊 Analyzing all survey categories using AMI structure (independent of selection)..."):
        categorized_df = get_unique_questions_by_category()
    
    if categorized_df.empty:
        st.markdown('<div class="warning-card">⚠️ No survey data available for categorization.</div>', unsafe_allow_html=True)
        st.markdown("**This page is now truly independent and will:**")
        st.markdown("• First try to load cached survey data")
        st.markdown("• If no cache exists, automatically fetch surveys from SurveyMonkey")
        st.markdown("• Process all available surveys for AMI structure categorization")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Force Refresh All Survey Data", use_container_width=True):
                # Clear all cached data and force fresh fetch
                st.session_state.all_questions = None
                st.session_state.dedup_questions = []
                st.session_state.dedup_choices = []
                st.session_state.fetched_survey_ids = []
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                st.rerun()
        
        with col2:
            if st.button("📋 Go to Survey Selection (Optional)", use_container_width=True):
                st.session_state.page = "survey_selection"
                st.rerun()
        st.stop()
    
    # Show data cleaning info
    st.markdown("### 🧹 Data Cleaning Applied")
    with st.expander("ℹ️ Question Text Cleaning Rules", expanded=False):
        st.markdown("**Automatic cleaning applied to questions:**")
        st.markdown("• Removed year specifications like `(i.e. 1 Jan. 2024 - 31 Dec. 2024)`")
        st.markdown("• For mobile number questions: Extracted main question `Your mobile number`")
        st.markdown("• Removed HTML formatting tags")
        st.markdown("• Grouped similar questions together")
        st.markdown("• **Example:** `Your mobile number <br>Country area code - +255 (Tanzania)` → `Your mobile number`")
    
    # AMI Structure metrics
    st.markdown("### 📊 AMI Structure Metrics")
    
    # Survey Stage metrics
    stage_stats = categorized_df.groupby('Survey Stage').agg({
        'question_text': 'count',
        'survey_count': 'sum'
    }).rename(columns={'question_text': 'question_count'}).reset_index()
    
    # Display metrics in columns for Survey Stages
    st.markdown("#### 📋 Survey Stages")
    cols = st.columns(min(len(stage_stats), 4))
    for idx, (_, row) in enumerate(stage_stats.iterrows()):
        with cols[idx % len(cols)]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                f"📋 {row['Survey Stage']}", 
                f"{row['question_count']} questions",
                f"From {row['survey_count']} surveys"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Respondent Type metrics
    resp_stats = categorized_df.groupby('Respondent Type').agg({
        'question_text': 'count',
        'survey_count': 'sum'
    }).rename(columns={'question_text': 'question_count'}).reset_index()
    
    st.markdown("#### 👥 Respondent Types")
    cols = st.columns(min(len(resp_stats), 3))
    for idx, (_, row) in enumerate(resp_stats.iterrows()):
        with cols[idx % len(cols)]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                f"👥 {row['Respondent Type']}", 
                f"{row['question_count']} questions",
                f"From {row['survey_count']} surveys"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Programme metrics
    prog_stats = categorized_df.groupby('Programme').agg({
        'question_text': 'count',
        'survey_count': 'sum'
    }).rename(columns={'question_text': 'question_count'}).reset_index()
    
    st.markdown("#### 🎓 Programmes")
    cols = st.columns(min(len(prog_stats), 4))
    for idx, (_, row) in enumerate(prog_stats.iterrows()):
        with cols[idx % len(cols)]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                f"🎓 {row['Programme']}", 
                f"{row['question_count']} questions",
                f"From {row['survey_count']} surveys"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    # AMI Structure filter and display
    st.markdown("### 🔍 Questions by AMI Structure")
    
    # Enhanced filters for AMI structure
    col1, col2, col3 = st.columns(3)
    with col1:
        survey_stage_filter = st.multiselect(
            "Filter by Survey Stage:",
            list(SURVEY_STAGES.keys()),
            default=list(SURVEY_STAGES.keys()),
            key="ami_survey_stage_filter"
        )
    
    with col2:
        respondent_type_filter = st.multiselect(
            "Filter by Respondent Type:",
            list(RESPONDENT_TYPES.keys()),
            default=list(RESPONDENT_TYPES.keys()),
            key="ami_respondent_type_filter"
        )
    
    with col3:
        programme_filter = st.multiselect(
            "Filter by Programme:",
            list(PROGRAMMES.keys()),
            default=list(PROGRAMMES.keys()),
            key="ami_programme_filter"
        )
    
    # Additional filters
    col1, col2, col3 = st.columns(3)
    with col1:
        schema_filter = st.multiselect(
            "Filter by question type:",
            ["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
            default=["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
            key="ami_schema_filter"
        )
    
    with col2:
        question_type_filter = st.selectbox(
            "Filter by question classification:",
            ["All", "Main Question", "Choice", "Heading"],
            index=0,
            key="ami_question_type_filter"
        )
    
    with col3:
        min_survey_count = st.number_input(
            "Minimum survey count:",
            min_value=1,
            value=1,
            key="ami_min_survey_count"
        )
    
    # Apply filters
    filtered_df = categorized_df.copy()
    
    if survey_stage_filter:
        filtered_df = filtered_df[filtered_df['Survey Stage'].isin(survey_stage_filter)]
    
    if respondent_type_filter:
        filtered_df = filtered_df[filtered_df['Respondent Type'].isin(respondent_type_filter)]
    
    if programme_filter:
        filtered_df = filtered_df[filtered_df['Programme'].isin(programme_filter)]
    
    if schema_filter:
        filtered_df = filtered_df[filtered_df['schema_type'].isin(schema_filter)]
    
    if min_survey_count > 1:
        filtered_df = filtered_df[filtered_df['survey_count'] >= min_survey_count]
    
    if question_type_filter != "All":
        if question_type_filter == "Main Question":
            filtered_df = filtered_df[filtered_df['is_choice'] == False]
            if 'question_category' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['question_category'] != "Heading"]
        elif question_type_filter == "Choice":
            filtered_df = filtered_df[filtered_df['is_choice'] == True]
        elif question_type_filter == "Heading":
            if 'question_category' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['question_category'] == "Heading"]
            else:
                filtered_df['temp_classification'] = filtered_df['question_text'].apply(classify_question)
                filtered_df = filtered_df[filtered_df['temp_classification'] == "Heading"]
                filtered_df = filtered_df.drop('temp_classification', axis=1)
    
    # UID Assignment Section
    if not filtered_df.empty:
        st.markdown("### 🔧 UID Assignment")
        
        # Prepare UID options
        uid_options = [None]
        if st.session_state.question_bank is not None and not st.session_state.question_bank.empty:
            uid_options.extend([f"{row['UID']} - {row['HEADING_0']}" for _, row in st.session_state.question_bank.iterrows()])
        
        # Display and edit questions with AMI structure
        display_columns = [
            "Survey Stage", "Respondent Type", "Programme", "question_text", "schema_type", 
            "is_choice", "survey_count", "Final_UID", "Change_UID", "required"
        ]
        
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        
        edited_categorized_df = st.data_editor(
            filtered_df[available_columns],
            column_config={
                "Survey Stage": st.column_config.TextColumn("Survey Stage", width="medium"),
                "Respondent Type": st.column_config.TextColumn("Respondent Type", width="medium"),
                "Programme": st.column_config.TextColumn("Programme", width="medium"),
                "question_text": st.column_config.TextColumn("Question/Choice", width="large"),
                "schema_type": st.column_config.TextColumn("Type", width="small"),
                "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
                "survey_count": st.column_config.NumberColumn("Survey Count", width="small"),
                "Final_UID": st.column_config.TextColumn("Current UID", width="medium"),
                "Change_UID": st.column_config.SelectboxColumn(
                    "Assign UID",
                    options=uid_options,
                    default=None,
                    width="large"
                ),
                "required": st.column_config.CheckboxColumn("Required", width="small")
            },
            disabled=["Survey Stage", "Respondent Type", "Programme", "question_text", "schema_type", "is_choice", "survey_count", "Final_UID"],
            hide_index=True,
            height=500,
            key="ami_categorized_editor"
        )
        
        # Process UID changes
        uid_changes_made = False
        for idx, row in edited_categorized_df.iterrows():
            if pd.notnull(row.get("Change_UID")):
                new_uid = row["Change_UID"].split(" - ")[0]
                categorized_df.at[idx, "Final_UID"] = new_uid
                categorized_df.at[idx, "configured_final_UID"] = new_uid
                uid_changes_made = True
        
        # Action buttons
        st.markdown("### 🚀 Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save UID Assignments", use_container_width=True):
                if uid_changes_made:
                    st.session_state.df_final = categorized_df.copy()
                    st.session_state.df_target = categorized_df.copy()
                    st.session_state.categorized_questions = categorized_df.copy()
                    st.markdown('<div class="success-card">✅ UID assignments saved successfully!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-card">⚠️ No UID changes to save.</div>', unsafe_allow_html=True)
        
        with col2:
            if st.button("🔧 Proceed to UID Matching", use_container_width=True):
                st.session_state.df_target = categorized_df.copy()
                st.session_state.page = "uid_matching"
                st.rerun()
        
        with col3:
            if st.button("📥 Export Category Data", use_container_width=True):
                csv_data = categorized_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    f"categorized_questions_{uuid4().hex[:8]}.csv",
                    "text/csv",
                    key="cat_download"
                )
        
        # Summary by AMI structure
        st.markdown("### 📊 Assignment Summary by AMI Structure")
        
        # Survey Stage summary
        stage_summary = categorized_df.groupby('Survey Stage').agg({
            'question_text': 'count',
            'Final_UID': lambda x: x.notna().sum()
        }).rename(columns={
            'question_text': 'Total Questions',
            'Final_UID': 'Assigned UIDs'
        })
        stage_summary['Assignment Rate %'] = (
            stage_summary['Assigned UIDs'] / stage_summary['Total Questions'] * 100
        ).round(2)
        
        st.markdown("#### 📋 By Survey Stage")
        st.dataframe(stage_summary, use_container_width=True)
        
        # Respondent Type summary
        resp_summary = categorized_df.groupby('Respondent Type').agg({
            'question_text': 'count',
            'Final_UID': lambda x: x.notna().sum()
        }).rename(columns={
            'question_text': 'Total Questions',
            'Final_UID': 'Assigned UIDs'
        })
        resp_summary['Assignment Rate %'] = (
            resp_summary['Assigned UIDs'] / resp_summary['Total Questions'] * 100
        ).round(2)
        
        st.markdown("#### 👥 By Respondent Type")
        st.dataframe(resp_summary, use_container_width=True)
        
        # Programme summary
        prog_summary = categorized_df.groupby('Programme').agg({
            'question_text': 'count',
            'Final_UID': lambda x: x.notna().sum()
        }).rename(columns={
            'question_text': 'Total Questions',
            'Final_UID': 'Assigned UIDs'
        })
        prog_summary['Assignment Rate %'] = (
            prog_summary['Assigned UIDs'] / prog_summary['Total Questions'] * 100
        ).round(2)
        
        st.markdown("#### 🎓 By Programme")
        st.dataframe(prog_summary, use_container_width=True)
        
    else:
        st.info("ℹ️ No questions match the selected filters")
    
    # Survey title analysis with AMI structure
    if st.expander("📋 Survey Title Analysis with AMI Structure", expanded=False):
        st.markdown("### 📊 How Surveys Were Categorized using AMI Structure")
        
        if st.session_state.all_questions is not None:
            survey_analysis = st.session_state.all_questions.groupby(['survey_title', 'survey_id']).first().reset_index()
            
            # Apply AMI structure categorization
            categorization_data = survey_analysis['survey_title'].apply(categorize_survey_by_ami_structure)
            categorization_df = pd.DataFrame(categorization_data.tolist())
            survey_analysis = pd.concat([survey_analysis, categorization_df], axis=1)
            
            # Display with enhanced AMI structure columns
            st.dataframe(
                survey_analysis[['survey_title', 'Survey Stage', 'Respondent Type', 'Programme', 'survey_id']],
                column_config={
                    "survey_title": st.column_config.TextColumn("Survey Title", width="large"),
                    "Survey Stage": st.column_config.TextColumn("Survey Stage", width="medium"),
                    "Respondent Type": st.column_config.TextColumn("Respondent Type", width="medium"),
                    "Programme": st.column_config.TextColumn("Programme", width="medium"),
                    "survey_id": st.column_config.TextColumn("Survey ID", width="small")
                },
                use_container_width=True,
                height=300
            )
            
            # Download survey analysis
            csv_survey_analysis = survey_analysis.to_csv(index=False)
            st.download_button(
                "📥 Download Survey Analysis",
                csv_survey_analysis,
                f"survey_ami_analysis_{uuid4().hex[:8]}.csv",
                "text/csv",
                key="survey_analysis_download"
            )

# ============= UID MATCHING PAGE =============
elif st.session_state.page == "uid_matching":
    st.markdown("## 🔧 UID Matching & Configuration")
    st.markdown('<div class="data-source-info">🔄 <strong>Process:</strong> Match survey questions → Snowflake references → Assign UIDs</div>', unsafe_allow_html=True)
    
    if st.session_state.df_target is None or st.session_state.df_target.empty:
        st.markdown('<div class="warning-card">⚠️ No survey data selected. Please select surveys first.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Go to Survey Selection"):
                st.session_state.page = "survey_selection"
                st.rerun()
        with col2:
            if st.button("📊 Go to AMI Categories"):
                st.session_state.page = "survey_categorization"
                st.rerun()
        st.stop()
    
    # Show survey data info
    st.markdown("### 📊 Current Survey Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Questions", len(st.session_state.df_target))
    with col2:
        main_q = len(st.session_state.df_target[st.session_state.df_target["is_choice"] == False])
        st.metric("Main Questions", main_q)
    with col3:
        if 'survey_id' in st.session_state.df_target.columns:
            surveys = st.session_state.df_target["survey_id"].nunique()
            st.metric("Surveys", surveys)
        else:
            st.metric("Surveys", "N/A")
    
    # Run UID Matching
    if st.session_state.df_final is None or st.button("🚀 Run UID Matching", type="primary"):
        try:
            with st.spinner("🔄 Running optimized UID matching (TF-IDF + Batch Semantic)..."):
                if st.session_state.question_bank is not None and not st.session_state.question_bank.empty:
                    # Show optimization info
                    st.info("⚡ **Performance Optimized:** Using pre-computed embeddings and batch processing for faster matching")
                    
                    # Create progress placeholder
                    progress_placeholder = st.empty()
                    progress_placeholder.write("📊 Phase 1: TF-IDF similarity calculation...")
                    
                    st.session_state.df_final = run_uid_match(st.session_state.question_bank, st.session_state.df_target)
                    
                    progress_placeholder.write("✅ Matching completed successfully!")
                    time.sleep(1)
                    progress_placeholder.empty()
                else:
                    st.session_state.df_final = st.session_state.df_target.copy()
                    st.session_state.df_final["Final_UID"] = None
        except Exception as e:
            st.markdown('<div class="warning-card">⚠️ UID matching failed. Continuing without UIDs.</div>', unsafe_allow_html=True)
            st.session_state.df_final = st.session_state.df_target.copy()
            st.session_state.df_final["Final_UID"] = None

    if st.session_state.df_final is not None:
        # Matching Results
        matched_percentage = calculate_matched_percentage(st.session_state.df_final)
        
        # Results Header
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Match Rate", f"{matched_percentage}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            high_conf = len(st.session_state.df_final[st.session_state.df_final.get("Match_Confidence", "") == "✅ High"])
            st.metric("✅ High Confidence", high_conf)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            low_conf = len(st.session_state.df_final[st.session_state.df_final.get("Match_Confidence", "") == "⚠️ Low"])
            st.metric("⚠️ Low Confidence", low_conf)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            no_match = len(st.session_state.df_final[st.session_state.df_final.get("Final_UID", pd.Series()).isna()])
            st.metric("❌ No Match", no_match)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔍 UID Matching Results")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            show_main_only = st.checkbox("Show main questions only", value=True)
        with col2:
            match_filter = st.multiselect(
                "Filter by match status:",
                ["✅ High", "⚠️ Low", "🧠 Semantic", "❌ No match"],
                default=["✅ High", "⚠️ Low", "🧠 Semantic"]
            )
        with col3:
            schema_filter = st.multiselect(
                "Filter by question type:",
                ["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
                default=["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"]
            )
        
        # Search
        search_query = st.text_input("🔍 Search questions/choices:")
        
        # Apply filters
        result_df = st.session_state.df_final.copy()
        if search_query:
            result_df = result_df[result_df["question_text"].str.contains(search_query, case=False, na=False)]
        if match_filter and "Final_Match_Type" in result_df.columns:
            result_df = result_df[result_df["Final_Match_Type"].isin(match_filter)]
        if show_main_only:
            result_df = result_df[result_df["is_choice"] == False]
        if schema_filter:
            result_df = result_df[result_df["schema_type"].isin(schema_filter)]
        
        # Configure UIDs
        if not result_df.empty:
            uid_options = [None]
            if st.session_state.question_bank is not None:
                uid_options.extend([f"{row['UID']} - {row['HEADING_0']}" for _, row in st.session_state.question_bank.iterrows()])
            
            # Create required column if it doesn't exist
            if "required" not in result_df.columns:
                result_df["required"] = False
            
            display_columns = ["question_uid", "question_text", "schema_type", "is_choice"]
            if "Final_UID" in result_df.columns:
                display_columns.append("Final_UID")
            if "Change_UID" not in result_df.columns:
                result_df["Change_UID"] = None
            display_columns.append("Change_UID")
            display_columns.append("required")
            
            # Only show columns that exist
            available_columns = [col for col in display_columns if col in result_df.columns]
            
            edited_df = st.data_editor(
                result_df[available_columns],
                column_config={
                    "question_uid": st.column_config.TextColumn("Question ID", width="medium"),
                    "question_text": st.column_config.TextColumn("Question/Choice", width="large"),
                    "schema_type": st.column_config.TextColumn("Type", width="medium"),
                    "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
                    "Final_UID": st.column_config.TextColumn("Current UID", width="medium"),
                    "Change_UID": st.column_config.SelectboxColumn(
                        "Change UID",
                        options=uid_options,
                        default=None,
                        width="large"
                    ),
                    "required": st.column_config.CheckboxColumn("Required", width="small")
                },
                disabled=["question_uid", "question_text", "schema_type", "is_choice", "Final_UID"],
                hide_index=True,
                height=400
            )
            
            # Apply UID changes
            for idx, row in edited_df.iterrows():
                if pd.notnull(row.get("Change_UID")):
                    new_uid = row["Change_UID"].split(" - ")[0]
                    st.session_state.df_final.at[idx, "Final_UID"] = new_uid
                    st.session_state.df_final.at[idx, "configured_final_UID"] = new_uid
                    st.session_state.uid_changes[idx] = new_uid
        
        # Export Section
        st.markdown("---")
        st.markdown("### 📥 Export & Upload")
        
        # Prepare export data - now returns two tables
        export_df_non_identity, export_df_identity = prepare_export_data(st.session_state.df_final)
        
        if not export_df_non_identity.empty or not export_df_identity.empty:
            
            # Show metrics for both tables
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Non-Identity Questions", len(export_df_non_identity))
            with col2:
                st.metric("🔐 Identity Questions", len(export_df_identity))
            with col3:
                total_records = len(export_df_non_identity) + len(export_df_identity)
                st.metric("📋 Total Records", total_records)
            
            # Preview both tables
            st.markdown("#### 👁️ Preview Data for Export")
            
            # Non-Identity Questions Preview
            if not export_df_non_identity.empty:
                st.markdown("**📊 Non-Identity Questions (Table 1)**")
                st.dataframe(export_df_non_identity.head(10), use_container_width=True)
            
            # Identity Questions Preview  
            if not export_df_identity.empty:
                st.markdown("**🔐 Identity Questions (Table 2)**")
                st.dataframe(export_df_identity.head(10), use_container_width=True)
            
            # Download options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not export_df_non_identity.empty:
                    csv_data_non_identity = export_df_non_identity.to_csv(index=False)
                    st.download_button(
                        "📥 Download Non-Identity CSV",
                        csv_data_non_identity,
                        f"non_identity_questions_{uuid4().hex[:8]}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if not export_df_identity.empty:
                    csv_data_identity = export_df_identity.to_csv(index=False)
                    st.download_button(
                        "📥 Download Identity CSV", 
                        csv_data_identity,
                        f"identity_questions_{uuid4().hex[:8]}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            with col3:
                if st.button("🚀 Upload Both Tables to Snowflake", use_container_width=True):
                    upload_to_snowflake_tables(export_df_non_identity, export_df_identity)

        else:
            st.warning("⚠️ No data available for export")

# ============= SURVEY CREATION PAGE =============
elif st.session_state.page == "survey_creation":
    st.markdown("## 🏗️ Survey Creation")
    st.markdown('<div class="data-source-info">🏗️ <strong>Process:</strong> Design survey → Configure questions → Deploy to SurveyMonkey</div>', unsafe_allow_html=True)
    
    with st.form("survey_creation_form"):
        st.markdown("### 📝 Survey Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            survey_title = st.text_input("Survey Title*", value="New Survey")
            survey_nickname = st.text_input("Survey Nickname", value=survey_title)
        with col2:
            survey_language = st.selectbox("Language", ["en", "es", "fr", "de"], index=0)
        
        st.markdown("### 📋 Questions")
        
        # Initialize edited_df in session state if it doesn't exist
        if "edited_df" not in st.session_state:
            st.session_state.edited_df = pd.DataFrame(columns=["question_text", "schema_type", "is_choice", "required"])

        edited_df = st.data_editor(
            st.session_state.edited_df,
            column_config={
                "question_text": st.column_config.SelectboxColumn(
                    "Question/Choice",
                    options=[""] + st.session_state.dedup_questions + st.session_state.dedup_choices,
                    default="",
                    width="large"
                ),
                "schema_type": st.column_config.SelectboxColumn(
                    "Question Type",
                    options=["Single Choice", "Multiple Choice", "Open-Ended", "Matrix"],
                    default="Open-Ended",
                    width="medium"
                ),
                "is_choice": st.column_config.CheckboxColumn("Is Choice", width="small"),
                "required": st.column_config.CheckboxColumn("Required", width="small")
            },
            hide_index=True,
            num_rows="dynamic",
            height=300
        )
        st.session_state.edited_df = edited_df

        # Validation and actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            validate_btn = st.form_submit_button("✅ Validate Questions", use_container_width=True)
        with col2:
            preview_btn = st.form_submit_button("👁️ Preview Survey", use_container_width=True)
        with col3:
            create_btn = st.form_submit_button("🚀 Create Survey", type="primary", use_container_width=True)
        
        # Process form submissions
        if validate_btn and st.session_state.question_bank is not None:
            non_standard = edited_df[~edited_df["question_text"].isin(st.session_state.question_bank["HEADING_0"])]
            if not non_standard.empty:
                st.markdown('<div class="warning-card">⚠️ Non-standard questions detected:</div>', unsafe_allow_html=True)
                st.dataframe(non_standard[["question_text"]], use_container_width=True)
                st.markdown("[📝 Submit New Questions](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
            else:
                st.markdown('<div class="success-card">✅ All questions are validated!</div>', unsafe_allow_html=True)
        
        if preview_btn or create_btn:
            if not survey_title or edited_df.empty:
                st.markdown('<div class="warning-card">⚠️ Survey title and questions are required.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">Survey creation functionality available</div>', unsafe_allow_html=True)

# ============= UNKNOWN PAGE HANDLER =============
else:
    st.error("❌ Unknown page requested")
    st.info("🏠 Redirecting to home...")
    st.session_state.page = "home"
    st.rerun()

# ============= FOOTER =============
st.markdown("---")

# Footer with quick links and status
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🔗 Quick Links**")
    st.markdown("📝 [Submit New Question](https://docs.google.com/forms/d/1LoY_La59UJ4ZsuxckM8Wl52kVeLI7a1t1MF8zIQxGUs)")
    st.markdown("🆔 [Submit New UID](https://docs.google.com/forms/d/1lkhfm1-t5-zwLxfbVEUiHewveLpGXv5yEVRlQx5XjxA)")

with footer_col2:
    st.markdown("**📊 Data Sources**")
    st.write("📊 SurveyMonkey: Surveys & Questions + IDs")
    st.write("❄️ Snowflake: UIDs & References")
    st.write("🎯 UID Final: Reference mappings")

with footer_col3:
    st.markdown("**📊 Current Session**")
    st.write(f"Page: {st.session_state.page}")
    st.write(f"SM Status: {'✅' if sm_status else '❌'}")
    st.write(f"SF Status: {'✅' if sf_status else '❌'}")
    uid_final_count = len(st.session_state.get('uid_final_reference', {}))
    st.write(f"UID Final: {uid_final_count}")
    
    # Show configured surveys count
    if sf_status and surveys:
        try:
            configured_count = count_configured_surveys_from_surveymonkey(surveys)
            st.write(f"Configured: {configured_count}")
        except:
            st.write("Configured: Error")
# ============= END OF SCRIPT =============



