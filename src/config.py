ROLE_SKILLS = {
    "Data Engineer": ["Python", "SQL", "Spark", "Hadoop", "ETL", "AWS", "Data Warehousing", "Airflow"],
    "Backend Developer": ["Java", "Python", "SQL", "REST API", "Spring Boot", "Git", "Docker", "Microservices"],
    "Cloud Engineer": ["Linux", "Networking", "AWS", "Docker", "Kubernetes", "Terraform", "Git", "Python"],
    "Data Scientist": ["Python", "SQL", "Statistics", "Machine Learning", "Pandas", "NumPy", "Scikit-learn", "Visualization"],
    "Full Stack Developer": ["JavaScript", "HTML", "CSS", "React", "Node.js", "SQL", "Git", "REST API"],
    "Cybersecurity Analyst": ["Networking", "Linux", "Cybersecurity", "Python", "SIEM", "SOC", "Cloud Security", "Incident Response"],
}

ALL_SKILLS = sorted({skill for skills in ROLE_SKILLS.values() for skill in skills} | {
    "C++", "MongoDB", "Firebase", "Power BI", "Tableau", "Flask", "Django", "FastAPI", "PostgreSQL"
})
