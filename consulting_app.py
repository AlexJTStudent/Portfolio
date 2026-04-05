# ============================================================
# Consulting DB — Streamlit Frontend App
# Covers:
#   1. INSERT a new consultant row (with live table refresh)
#   2. Query: consultants + their jobs (inner join + WHERE filter)
# ============================================================

import streamlit as st
import mysql.connector

st.set_page_config(page_title="Consulting DB", layout="centered")
st.title("🏢 Consulting Database Manager")

# ──────────────────────────────────────────────
# CONNECTION SETTINGS — update password if needed
# ──────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 3306
DB   = "Consulting_db"
USER = "root"
PWD  = "itm222SQLpw@"      # ← change to your local MySQL password if different


def get_conn():
    return mysql.connector.connect(
        host=HOST, port=PORT, user=USER, password=PWD, database=DB
    )


def run_query(sql, params=()):
    """Run a SELECT and return a list of dicts."""
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def run_insert(sql, params=()):
    """Run an INSERT / UPDATE / DELETE (commits automatically)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    cur.close()
    conn.close()


def load_lookup(table, id_col, name_col):
    """Helper: fetch id→name pairs for a dropdown."""
    rows = run_query(f"SELECT {id_col}, {name_col} FROM {table} ORDER BY {name_col}")
    return {row[name_col]: row[id_col] for row in rows}


# ════════════════════════════════════════════════════════════
# SECTION 1 — INSERT A NEW CONSULTANT
# ════════════════════════════════════════════════════════════
st.header("➕ Add a New Consultant")
st.write(
    "Fill in the fields below and click **Add Consultant** to insert a new row "
    "into the `consultant` table."
)

with st.form("insert_consultant"):
    col1, col2 = st.columns(2)
    with col1:
        first_name = st.text_input("First Name")
        skill_level = st.selectbox(
            "Skill Level", ["Junior", "Mid-Level", "Senior", "Principal"]
        )
    with col2:
        last_name = st.text_input("Last Name")
        availability = st.selectbox(
            "Availability Status", ["Available", "Busy", "On Leave"]
        )

    # Dropdowns populated from the DB
    try:
        roles = load_lookup("role", "role_id", "role_name")
        industries = load_lookup("industry", "industry_id", "specialization")
        role_choice = st.selectbox("Role", list(roles.keys()))
        industry_choice = st.selectbox("Industry Specialization", list(industries.keys()))
    except Exception as e:
        st.error(f"Could not load lookup tables: {e}")
        roles, industries = {}, {}
        role_choice = industry_choice = None

    submitted = st.form_submit_button("Add Consultant")

if submitted:
    if not first_name.strip() or not last_name.strip():
        st.warning("Please enter both first and last name.")
    elif not roles or not industries:
        st.error("Cannot insert — lookup tables failed to load.")
    else:
        try:
            run_insert(
                """
                INSERT INTO consultant
                    (first_name, last_name, skill_level,
                     availability_status, role_role_id, industry_industry_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    first_name.strip(),
                    last_name.strip(),
                    skill_level,
                    availability,
                    roles[role_choice],
                    industries[industry_choice],
                ),
            )
            st.success(
                f"✅ Consultant **{first_name} {last_name}** added successfully!"
            )
        except Exception as e:
            st.error(f"Insert failed: {e}")

# Show current consultant table to prove the insert worked
st.subheader("Current Consultants")
try:
    consultants = run_query(
        """
        SELECT
            c.consultant_id   AS ID,
            c.first_name      AS `First Name`,
            c.last_name       AS `Last Name`,
            c.skill_level     AS `Skill Level`,
            c.availability_status AS `Status`,
            r.role_name       AS Role,
            i.specialization  AS Industry
        FROM consultant c
        INNER JOIN role     r ON c.role_role_id        = r.role_id
        INNER JOIN industry i ON c.industry_industry_id = i.industry_id
        ORDER BY c.consultant_id DESC
        """
    )
    if consultants:
        st.dataframe(consultants, use_container_width=True)
        st.caption(f"{len(consultants)} consultant(s) on record.")
    else:
        st.info("No consultants found yet.")
except Exception as e:
    st.error(f"Could not load consultants: {e}")

st.divider()

# ════════════════════════════════════════════════════════════
# SECTION 2 — QUERY: Consultants assigned to jobs (Inner Join)
# with a WHERE filter on job status
# ════════════════════════════════════════════════════════════
st.header("🔍 Query: Consultants by Job Status")
st.write(
    "This query uses an **INNER JOIN** across `consultant`, `job_has_consultant`, "
    "and `job` to show which consultants are working on jobs that match a given "
    "status (e.g. *In Progress*, *Completed*, *Pending*)."
)

SAMPLE_SQL = """
SELECT
    c.consultant_id              AS `Consultant ID`,
    CONCAT(c.first_name, ' ', c.last_name) AS `Consultant Name`,
    c.skill_level                AS `Skill Level`,
    j.job_id                     AS `Job ID`,
    j.description                AS `Job Description`,
    j.priority_level             AS Priority,
    j.job_status                 AS `Job Status`,
    j.end_deadline               AS Deadline
FROM consultant c
INNER JOIN job_has_consultant jhc ON c.consultant_id = jhc.consultant_consultant_id
INNER JOIN job j                  ON jhc.job_job_id  = j.job_id
WHERE j.job_status LIKE %s
ORDER BY j.priority_level, c.last_name
LIMIT 100;
"""

status_filter = st.text_input(
    "Filter by Job Status (e.g. In Progress, Completed, Pending — leave blank for all)",
    value=""
)

if st.button("Run Query"):
    try:
        param = f"%{status_filter.strip()}%"
        rows = run_query(SAMPLE_SQL, (param,))
        if rows:
            st.dataframe(rows, use_container_width=True)
            st.success(f"Returned {len(rows)} row(s).")
        else:
            st.info("No rows matched. Try a different status or leave the field blank.")
    except Exception as e:
        st.error(f"Query failed: {e}")

st.divider()
st.caption("ITM 220 Portfolio Project · Consulting_db")
