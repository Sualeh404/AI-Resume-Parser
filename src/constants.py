"""
constants.py

Classification sets used across the pipeline. Everything here came from
scanning the full 100,000-candidate file with explore.py — not from
guessing or sampling. The sample is useful for a first look but misses
rare titles. On the first pass this list was hand-picked from a 50-row
sample and came up 43 candidates short, all senior/staff/lead variants.
"""

# All current_title values in the dataset that map to genuine AI/ML work.
# 1,179 candidates total across the full pool.
AI_TITLES = {
    "ML Engineer",
    "AI Research Engineer",
    "Data Scientist",
    "Senior Software Engineer (ML)",
    "Computer Vision Engineer",
    "Junior ML Engineer",
    "AI Specialist",
    "Recommendation Systems Engineer",
    "Machine Learning Engineer",
    "Applied ML Engineer",
    "Search Engineer",
    "AI Engineer",
    "Senior Data Scientist",
    "NLP Engineer",
    "Senior NLP Engineer",
    "Senior Machine Learning Engineer",
    "Staff Machine Learning Engineer",
    "Senior AI Engineer",
    "Senior Applied Scientist",
    "Lead AI Engineer",
}

# The JD specifically calls out CV/speech/robotics backgrounds as risky:
# someone whose only experience is computer vision would be re-learning
# fundamentals for a search/retrieval role. These titles still pass the
# AI_TITLES check but get a partial penalty unless their career history
# shows real NLP/IR work.
CV_SPEECH_TITLES = {
    "Computer Vision Engineer",
}

# Real, currently-operating product companies present in the dataset.
# The JD rewards product-company experience over pure IT services.
#
# First version of this list was built from the top-30 frequency count
# across the full 100k pool and missed 29 companies that appear only
# in the AI-titled pool — global big tech and Indian AI startups.
# Found by catching nonsense reasoning output ("Meta is a smaller,
# unlisted firm") then tracing it back here.
REAL_PRODUCT_COMPANIES = {
    # Indian consumer/startup
    "Swiggy", "CRED", "Razorpay", "Zomato", "Flipkart", "Meesho",
    "Zoho", "Freshworks", "InMobi", "Nykaa", "Vedantu", "Ola",
    "Paytm", "PolicyBazaar", "Unacademy",
    # Global big tech
    "Meta", "Google", "Amazon", "Microsoft", "Apple", "Adobe",
    "Netflix", "Salesforce", "LinkedIn", "Uber",
    # Indian AI startups found specifically in the AI-titled pool
    "Sarvam AI", "Krutrim", "Wysa", "Yellow.ai", "Haptik", "Niramai",
    "Aganitha", "Glance", "Rephrase.ai", "Saarthi.ai", "Verloop.io",
    "Mad Street Den", "Dream11", "BYJU'S", "Locobuzz", "PharmEasy",
    "Observe.AI", "PhonePe", "upGrad",
}

# IT services / consulting firms the JD explicitly down-weights.
# Important: being currently at one of these is NOT a disqualifier
# if the candidate also has prior product-company experience.
IT_SERVICES_COMPANIES = {
    "TCS", "Infosys", "Wipro", "Cognizant", "Capgemini", "Accenture",
    "HCL", "Tech Mahindra", "Mphasis", "Mindtree", "Genpact AI",
}

# Fictional filler companies from Office/Silicon Valley/Marvel.
# About 60k of the 100k candidates work at one of these.
# No positive or negative signal — just noise.
FICTIONAL_FILLER_COMPANIES = {
    "Wayne Enterprises", "Stark Industries", "Hooli", "Pied Piper",
    "Globex Inc", "Acme Corp", "Dunder Mifflin", "Initech",
}

# Language patterns that let a non-AI-titled candidate get promoted
# into scoring if their career history shows real retrieval/ranking
# ownership. As of the current dataset, zero candidates trigger this —
# the data's strongest signal is entirely inside the AI-titled pool.
# Kept because the JD explicitly asks for this reasoning and future
# versions of the data might look different.
OVERRIDE_PATTERNS = [
    r"migrat\w+ .{0,40}(embedding|retrieval|search)",
    r"owned? .{0,40}(ranking|retrieval|recommendation|matching) (system|model|engine|pipeline)",
    r"(built|design\w*|shipped|architected) .{0,40}(ranking|retrieval|recommendation) (system|model|engine)",
    r"(led|drove) .{0,40}(migration|transition) .{0,40}(embedding|retrieval|vector)",
]

# JD-preferred locations. Matched as substrings against the location field.
JD_PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "bangalore", "bengaluru",
}

# Small conservative set of AI skill names used only by explore.py for
# quick counts and sanity checks. Not used in actual scoring.
AI_SKILL_NAMES = {
    "NLP", "Fine-tuning LLMs", "LoRA", "Speech Recognition", "Image Classification",
    "GANs", "Prompt Engineering", "Haystack", "Kubeflow", "Weights & Biases",
    "Milvus", "TTS", "BentoML", "Feature Engineering",
}
