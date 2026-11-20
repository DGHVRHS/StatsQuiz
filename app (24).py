import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import urllib.request
import json
import smtplib
from email.message import EmailMessage
from datetime import datetime

st.set_page_config(page_title="Stats Regression Quiz", layout="wide")

TEACHER_EMAIL = "dgreene@region1schools.org"

# Scenarios modeled directly on Stats Homework #9
SCENARIOS = [
    {
        "title": "Study Habits & Exam Performance",
        "x_name": "Hours Studied",
        "x_unit": "hours",
        "y_name": "Exam Score",
        "y_unit": "points",
        "x_min": 1.0,
        "x_max": 9.5,
        "true_a": 52.0,
        "true_b": 4.6,
        "noise": 2.8,
        "y_bounds": (0, 100),
        "conv_text": "converting Exam Score from points (out of 100) to a 4.0 GPA scale using G = 0.04y",
        "outlier_desc": "a student who studied 9.0 hours but scored only 42 points due to severe test anxiety",
        "yint_meaningful": True,
        "yint_reason": "Yes — 0 hours studied is possible and represents the predicted baseline exam score with no studying.",
        "extrap_x": 18.0,
        "extrap_issue": "predicting for 18 hours is far outside the observed data (extrapolation) and predicts an impossible exam score over 100%."
    },
    {
        "title": "Urban Tree Canopy & Summer Heat",
        "x_name": "Tree Canopy Cover",
        "x_unit": "%",
        "y_name": "Avg Summer Afternoon Temp",
        "y_unit": "°F",
        "x_min": 8.0,
        "x_max": 55.0,
        "true_a": 94.5,
        "true_b": -0.34,
        "noise": 1.4,
        "y_bounds": (60, 105),
        "conv_text": "converting temperature from degrees Fahrenheit to degrees Celsius using C = (5/9)(F - 32)",
        "outlier_desc": "an industrial neighborhood with 50% tree cover that still had an unusually high temperature of 93°F due to a power plant",
        "yint_meaningful": True,
        "yint_reason": "Yes — 0% tree canopy cover is possible and represents the predicted summer temperature in a neighborhood with no trees.",
        "extrap_x": 95.0,
        "extrap_issue": "predicting for 95% canopy cover is far outside the observed range of neighborhoods (extrapolation)."
    },
    {
        "title": "Café Temperature & Iced Latte Sales",
        "x_name": "Daily High Temperature",
        "x_unit": "°F",
        "y_name": "Iced Lattes Sold",
        "y_unit": "lattes",
        "x_min": 50.0,
        "x_max": 88.0,
        "true_a": -34.0,
        "true_b": 2.05,
        "noise": 5.5,
        "y_bounds": (20, 200),
        "conv_text": "converting Daily High Temperature from °F to °C using C = (5/9)(F - 32)",
        "outlier_desc": "a cool 55°F day where the café sold 145 iced lattes because of a local festival",
        "yint_meaningful": False,
        "yint_reason": "No — predicting negative iced lattes sold at 0°F is impossible in context and far outside the observed temperatures.",
        "extrap_x": 115.0,
        "extrap_issue": "predicting sales at 115°F is far outside the observed temperature range (extrapolation)."
    },
    {
        "title": "Strength Training & 100-Meter Dash Time",
        "x_name": "Weekly Strength Training",
        "x_unit": "hours",
        "y_name": "100-Meter Dash Time",
        "y_unit": "seconds",
        "x_min": 1.0,
        "x_max": 8.0,
        "true_a": 13.85,
        "true_b": -0.26,
        "noise": 0.16,
        "y_bounds": (9.8, 15.0),
        "conv_text": "converting Weekly Strength Training from hours to minutes (Minutes = 60 × Hours)",
        "outlier_desc": "a new athlete who trained 15 hours per week and ran a slow 14.5-second dash due to overtraining fatigue",
        "yint_meaningful": True,
        "yint_reason": "Yes — 0 hours of weekly strength training is possible and represents the predicted sprint time for an untrained runner.",
        "extrap_x": 25.0,
        "extrap_issue": "predicting sprint time at 25 hours/week is far outside the observed 1–8 hour range (extrapolation) and would eventually predict impossibly fast times."
    }
]


def generate_simulation():
    rng = np.random.default_rng()
    scen = SCENARIOS[int(rng.integers(0, len(SCENARIOS)))]
    n = 10

    while True:
        x_base = np.linspace(scen["x_min"], scen["x_max"], n)
        x_jitter = rng.normal(0, (scen["x_max"] - scen["x_min"]) * 0.03, size=n)
        x_vals = np.round(np.clip(x_base + x_jitter, scen["x_min"], scen["x_max"]), 1)
        x_vals = np.sort(x_vals)

        y_raw = scen["true_a"] + scen["true_b"] * x_vals + rng.normal(0, scen["noise"], size=n)
        y_vals = np.round(np.clip(y_raw, scen["y_bounds"][0], scen["y_bounds"][1]), 1)

        if len(np.unique(x_vals)) >= 8 and len(np.unique(y_vals)) >= 8:
            r_check = float(np.corrcoef(x_vals, y_vals)[0, 1])
            if 0.76 <= abs(r_check) <= 0.96:
                break

    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    slope = float(slope)
    intercept = float(intercept)
    r = float(np.corrcoef(x_vals, y_vals)[0, 1])

    y_pred = intercept + slope * x_vals
    residuals = y_vals - y_pred
    target_idx = int(np.argmax(np.abs(residuals)))

    df = pd.DataFrame({
        "Observation": [f"Obs {i+1}" for i in range(n)],
        scen["x_name"]: x_vals,
        scen["y_name"]: y_vals
    })

    return {
        "scenario": scen,
        "df": df,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "r": r,
        "r_sq": r ** 2,
        "slope": slope,
        "intercept": intercept,
        "target_idx": target_idx
    }


def send_results_email(student_name, student_email, score, total, report_text):
    subject = f"Stats Quiz Submission: {student_name} ({score}/{total})"

    # Option 1: Direct SMTP if configured in Streamlit Secrets
    if "smtp_user" in st.secrets and "smtp_password" in st.secrets:
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = st.secrets["smtp_user"]
            msg["To"] = TEACHER_EMAIL
            if student_email.strip():
                msg["Reply-To"] = student_email.strip()
            msg.set_content(report_text)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=12) as server:
                server.login(st.secrets["smtp_user"], st.secrets["smtp_password"])
                server.send_message(msg)
            return True, "Sent directly via SMTP."
        except Exception as e:
            return False, f"SMTP Error: {e}"

    # Option 2: Zero-config webhook via FormSubmit
    try:
        url = f"https://formsubmit.co/ajax/{TEACHER_EMAIL}"
        payload = {
            "_subject": subject,
            "Student Name": student_name,
            "Student Email": student_email or "Not provided",
            "Quiz Score": f"{score} / {total} ({round((score / total) * 100)}%)",
            "Submission Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Detailed Results": report_text
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
            if resp_body.get("success") in [True, "true"]:
                return True, resp_body.get("message", "Email dispatched.")
            return False, str(resp_body)
    except Exception as e:
        return False, str(e)


# --- INITIALIZE SESSION STATE ---
if "sim" not in st.session_state:
    st.session_state.sim = generate_simulation()
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "grade_report" not in st.session_state:
    st.session_state.grade_report = None

sim = st.session_state.sim
scen = sim["scenario"]
r = sim["r"]
slope = sim["slope"]
intercept = sim["intercept"]

sign_str = "+" if slope >= 0 else "-"
eq_str = f"ŷ = {intercept:.3f} {sign_str} {abs(slope):.3f}x"

# --- HEADER ---
st.title("📝 Simulated Linear Regression & Correlation Quiz")
st.markdown(
    f"Analyze your unique simulated dataset below, answer all **7 questions**, "
    f"and submit your work to email your results directly to **{TEACHER_EMAIL}**."
)

top_c1, top_c2, top_c3 = st.columns([1.5, 1.5, 1])
with top_c1:
    student_name = st.text_input("Student Full Name (Required):", placeholder="Enter your first and last name")
with top_c2:
    student_email = st.text_input("Your School Email (Optional):", placeholder="name@region1schools.org")
with top_c3:
    st.write("")
    st.write("")
    if st.button("🔄 New Simulated Dataset", disabled=st.session_state.submitted):
        st.session_state.sim = generate_simulation()
        st.rerun()

# --- STEP 1: SIMULATED DATA & STATISTICAL ANALYSIS ---
st.divider()
st.header(f"Part 1: Simulated Dataset — {scen['title']}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Correlation (r)", f"{r:.4f}")
m2.metric("R-Squared (r²)", f"{sim['r_sq']:.4f}")
m3.metric("Slope (b)", f"{slope:.3f}")
m4.metric("y-Intercept (a)", f"{intercept:.3f}")

st.info(f"**Least-Squares Regression Line (LSRL):**  `{eq_str}`   |   **x:** {scen['x_name']} ({scen['x_unit']})   |   **y:** {scen['y_name']} ({scen['y_unit']})")

col_chart, col_table = st.columns([1.45, 1])

with col_chart:
    plot_df = sim["df"].copy()
    line_df = pd.DataFrame({
        scen["x_name"]: [float(sim["x_vals"].min()), float(sim["x_vals"].max())],
        scen["y_name"]: [
            intercept + slope * float(sim["x_vals"].min()),
            intercept + slope * float(sim["x_vals"].max())
        ]
    })

    scatter = (
        alt.Chart(plot_df)
        .mark_circle(size=95, color="#0f766e")
        .encode(
            x=alt.X(f"{scen['x_name']}:Q", title=f"{scen['x_name']} ({scen['x_unit']})", scale=alt.Scale(zero=False)),
            y=alt.Y(f"{scen['y_name']}:Q", title=f"{scen['y_name']} ({scen['y_unit']})", scale=alt.Scale(zero=False)),
            tooltip=["Observation", scen["x_name"], scen["y_name"]]
        )
    )

    reg_line = (
        alt.Chart(line_df)
        .mark_line(color="#1d4ed8", strokeWidth=3)
        .encode(
            x=f"{scen['x_name']}:Q",
            y=f"{scen['y_name']}:Q"
        )
    )

    st.altair_chart(
        (scatter + reg_line).properties(title=f"{scen['y_name']} vs. {scen['x_name']} (n = 10)", height=360),
        use_container_width=True
    )

with col_table:
    st.subheader("Observed Data Table")
    st.dataframe(sim["df"], use_container_width=True, height=360, hide_index=True)

# --- STEP 2: 7 WORKSHEET-BASED QUESTIONS ---
st.divider()
st.header("Part 2: Quiz Questions (7 Questions)")

true_dir = "Positive" if r > 0 else "Negative"
true_str = "Strong" if abs(r) >= 0.70 else ("Moderate" if abs(r) >= 0.40 else "Weak")

# Q1: Direction, Strength & Contextual Interpretation
with st.container(border=True):
    st.subheader("Question 1: Direction, Strength & Interpreting r")
    q1_opts = [
        f"There is a strong, {true_dir.lower()} linear relationship between {scen['x_name'].lower()} and {scen['y_name'].lower()}.",
        f"There is a weak, {true_dir.lower()} linear relationship between {scen['x_name'].lower()} and {scen['y_name'].lower()}.",
        f"There is a strong, {'negative' if r > 0 else 'positive'} linear relationship between {scen['x_name'].lower()} and {scen['y_name'].lower()}.",
        f"Changes in {scen['x_name'].lower()} directly cause {round(abs(r)*100, 1)}% of the changes in {scen['y_name'].lower()}."
    ]
    ans_q1 = st.radio(
        f"Based on your scatterplot and correlation coefficient r = {r:.4f}, which statement accurately interprets r in context?",
        q1_opts,
        index=None,
        key="q1",
        disabled=st.session_state.submitted
    )

# Q2: Properties of r & Unit Conversions
with st.container(border=True):
    st.subheader("Question 2: Unit Conversions & Properties of r")
    q2_opts = [
        f"r = {r:.4f} (it remains completely unchanged because correlation is unitless and unaffected by linear transformations).",
        f"r changes in proportion to the conversion scale and takes on units of {scen['y_unit']} per {scen['x_unit']}.",
        f"r = {-r:.4f} (the sign flips whenever constants are subtracted or multiplied).",
        "r moves closer to 1.00 because larger unit values increase the strength of the linear fit."
    ]
    ans_q2 = st.radio(
        f"Suppose you transform your dataset by {scen['conv_text']}. What will the new correlation coefficient r be?",
        q2_opts,
        index=None,
        key="q2",
        disabled=st.session_state.submitted
    )

# Q3: Effect of Outliers on Correlation
with st.container(border=True):
    st.subheader("Question 3: Outliers & Influential Observations")
    q3_opts = [
        "r would move closer to 0 (the linear correlation would weaken) because the outlier falls far away from the overall linear trend.",
        "r would move closer to +1 or -1 (the linear correlation would strengthen) because outliers pull the regression line tighter.",
        "r would stay exactly the same because outliers only affect the y-intercept, not correlation.",
        "r would become greater than +1.0 or less than -1.0."
    ]
    ans_q3 = st.radio(
        f"Suppose a new data point is added to your dataset representing {scen['outlier_desc']}. How would adding this outlier affect r = {r:.4f}?",
        q3_opts,
        index=None,
        key="q3",
        disabled=st.session_state.submitted
    )

# Q4: Association vs. Causation
with st.container(border=True):
    st.subheader("Question 4: Association vs. Causation")
    q4_opts = [
        f"No — a strong correlation in observational data shows association, not causation, and confounding variables may explain the relationship.",
        f"Yes — because |r| = {abs(r):.4f} is above 0.70, changing {scen['x_name'].lower()} is guaranteed to cause {scen['y_name'].lower()} to change.",
        "Yes — whenever a least-squares regression line has a non-zero slope, x must be the direct cause of y.",
        "No — correlation can only prove causation when r is negative."
    ]
    ans_q4 = st.radio(
        f"Based purely on the strong correlation (r = {r:.4f}) in this observational dataset, can we conclude that changing {scen['x_name']} directly causes {scen['y_name']} to change?",
        q4_opts,
        index=None,
        key="q4",
        disabled=st.session_state.submitted
    )

# Q5: Interpreting Slope & y-Intercept
with st.container(border=True):
    st.subheader("Question 5: Interpreting Slope & y-Intercept")
    inc_dec = "increases" if slope >= 0 else "decreases"
    q5_opts = [
        f"For each additional 1 {scen['x_unit']} of {scen['x_name']}, the predicted {scen['y_name']} {inc_dec} by {abs(slope):.3f} {scen['y_unit']}.",
        f"For each additional 1 {scen['y_unit']} of {scen['y_name']}, the predicted {scen['x_name']} {inc_dec} by {abs(slope):.3f} {scen['x_unit']}.",
        f"When {scen['x_name']} is 0 {scen['x_unit']}, the predicted {scen['y_name']} is {abs(slope):.3f} {scen['y_unit']}.",
        f"Every single observation's {scen['y_name']} changes by exactly {abs(slope):.3f} {scen['y_unit']} per {scen['x_unit']}."
    ]
    ans_q5 = st.radio(
        f"Which statement properly interprets the slope ({slope:.3f}) of your regression equation `{eq_str}` in context?",
        q5_opts,
        index=None,
        key="q5",
        disabled=st.session_state.submitted
    )

# Q6: Prediction & Residual Calculation
t_idx = sim["target_idx"]
t_obs = sim["df"].loc[t_idx, "Observation"]
t_x = float(sim["x_vals"][t_idx])
t_y = float(sim["y_vals"][t_idx])
t_yhat_exact = intercept + slope * t_x
t_yhat_disp = round(intercept, 3) + round(slope, 3) * t_x
t_res_exact = t_y - t_yhat_exact
t_res_disp = t_y - t_yhat_disp

with st.container(border=True):
    st.subheader("Question 6: Regression Prediction & Calculating a Residual")
    st.markdown(
        f"Consider **{t_obs}** from your data table: **({scen['x_name']} = {t_x}, {scen['y_name']} = {t_y})**."
    )
    q6_c1, q6_c2 = st.columns(2)
    with q6_c1:
        ans_q6_yhat = st.number_input(
            f"6a. Using `{eq_str}`, compute the predicted {scen['y_name']} (ŷ) when x = {t_x}:",
            value=0.0,
            format="%.2f",
            key="q6_yhat",
            disabled=st.session_state.submitted
        )
    with q6_c2:
        ans_q6_res = st.number_input(
            f"6b. Compute the residual (Residual = Actual y − Predicted ŷ):",
            value=0.0,
            format="%.2f",
            key="q6_res",
            disabled=st.session_state.submitted
        )

# Q7: Extrapolation
with st.container(border=True):
    st.subheader("Question 7: Reliability of Predictions & Extrapolation")
    q7_opts = [
        f"Unreliable — using a regression line to predict at x = {scen['extrap_x']} {scen['x_unit']} is called extrapolation because it lies far outside the observed x-range.",
        f"Reliable — linear regression lines remain accurate for all real numbers regardless of the observed domain.",
        f"Unreliable — using a regression line outside the observed range is called confounding.",
        f"Reliable — as long as r² > 0.50, predictions far beyond the data have zero residual error."
    ]
    ans_q7 = st.radio(
        f"Suppose someone uses `{eq_str}` to predict {scen['y_name']} when {scen['x_name']} = {scen['extrap_x']} {scen['x_unit']}. Is this prediction reliable, and what is the statistical term for it?",
        q7_opts,
        index=None,
        key="q7",
        disabled=st.session_state.submitted
    )

# --- STEP 3: SUBMIT & EMAIL RESULTS ---
st.divider()
st.header("Step 3: Submit Quiz & Email Results")

if not st.session_state.submitted:
    if st.button("📨 Grade & Email My Quiz Results", type="primary", use_container_width=True):
        if not student_name.strip():
            st.error("⚠️ Please enter your **Full Name** at the top of the page before submitting.")
        elif None in [ans_q1, ans_q2, ans_q3, ans_q4, ans_q5, ans_q7]:
            st.error("⚠️ Please answer all multiple-choice questions (Questions 1–5 and Question 7) before submitting.")
        else:
            # Grade all 7 questions
            q1_ok = (ans_q1 == q1_opts[0])
            q2_ok = (ans_q2 == q2_opts[0])
            q3_ok = (ans_q3 == q3_opts[0])
            q4_ok = (ans_q4 == q4_opts[0])
            q5_ok = (ans_q5 == q5_opts[0])

            yhat_ok = min(abs(ans_q6_yhat - t_yhat_exact), abs(ans_q6_yhat - t_yhat_disp)) <= 0.25
            res_ok = min(abs(ans_q6_res - t_res_exact), abs(ans_q6_res - t_res_disp)) <= 0.25
            q6_ok = yhat_ok and res_ok

            q7_ok = (ans_q7 == q7_opts[0])

            results_list = [
                ("Q1 (Direction, Strength & Context)", q1_ok, ans_q1, q1_opts[0]),
                ("Q2 (Unit Conversion & Properties of r)", q2_ok, ans_q2, q2_opts[0]),
                ("Q3 (Effect of Outliers on r)", q3_ok, ans_q3, q3_opts[0]),
                ("Q4 (Association vs. Causation)", q4_ok, ans_q4, q4_opts[0]),
                ("Q5 (Slope Interpretation)", q5_ok, ans_q5, q5_opts[0]),
                (
                    "Q6 (Prediction & Residual)",
                    q6_ok,
                    f"ŷ = {ans_q6_yhat:.2f}, Residual = {ans_q6_res:.2f}",
                    f"ŷ = {t_yhat_disp:.2f}, Residual = {t_res_disp:.2f}"
                ),
                ("Q7 (Extrapolation)", q7_ok, ans_q7, q7_opts[0]),
            ]

            score = sum(1 for _, ok, _, _ in results_list if ok)
            total = len(results_list)
            pct = round((score / total) * 100)

            report_lines = [
                f"STATS REGRESSION QUIZ RESULTS",
                f"Student: {student_name.strip()}",
                f"Student Email: {student_email.strip() or 'N/A'}",
                f"Score: {score} / {total} ({pct}%)",
                f"Scenario: {scen['title']}",
                f"Simulated Stats: r = {r:.4f}, r^2 = {sim['r_sq']:.4f}, LSRL: {eq_str}",
                f"Target Observation ({t_obs}): x = {t_x}, y = {t_y}",
                "-" * 50
            ]
            for label, ok, chosen, correct in results_list:
                status = "CORRECT" if ok else "INCORRECT"
                report_lines.append(f"{label}: [{status}]")
                report_lines.append(f"   Student Answer: {chosen}")
                if not ok:
                    report_lines.append(f"   Correct Answer: {correct}")

            report_text = "\n".join(report_lines)

            with st.spinner(f"Emailing results to {TEACHER_EMAIL}..."):
                email_sent, email_msg = send_results_email(
                    student_name.strip(), student_email, score, total, report_text
                )

            st.session_state.submitted = True
            st.session_state.grade_report = {
                "score": score,
                "total": total,
                "pct": pct,
                "results_list": results_list,
                "report_text": report_text,
                "email_sent": email_sent,
                "email_msg": email_msg
            }
            st.rerun()

# --- DISPLAY FINAL RESULTS AFTER SUBMISSION ---
if st.session_state.submitted and st.session_state.grade_report:
    rep = st.session_state.grade_report
    st.balloons()

    if rep["email_sent"]:
        st.success(f"✅ **Submitted!** Your score of **{rep['score']}/{rep['total']} ({rep['pct']}%)** has been emailed to **{TEACHER_EMAIL}**.")
    else:
        st.warning(
            f"⚠️ Quiz graded (**{rep['score']}/{rep['total']} — {rep['pct']}%**), but automatic email delivery encountered a network block ({rep['email_msg']}). "
            f"Click the backup email link below or download your receipt to send to **{TEACHER_EMAIL}**."
        )

    st.subheader(f"Final Score: {rep['score']} / {rep['total']} ({rep['pct']}%)")
    for label, ok, chosen, correct in rep["results_list"]:
        if ok:
            st.markdown(f"✅ **{label}** — Correct")
        else:
            st.markdown(f"❌ **{label}** — Your answer: *{chosen}* | Correct: *{correct}*")

    st.download_button(
        label="📄 Download Score Receipt (.txt)",
        data=rep["report_text"],
        file_name=f"Stats_Quiz_{student_name.strip().replace(' ', '_')}.txt",
        mime="text/plain"
    )
