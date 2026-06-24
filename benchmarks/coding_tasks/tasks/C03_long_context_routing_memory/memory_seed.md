# Long-context routing archive for C03

This memory is intentionally long. It models the kind of messy cross-session policy memo that is expensive to paste into every coding prompt. Most sections are operational background, incident notes, and non-routing context. The useful routing table is buried in the middle.

Section 01 - archive scope. The support organization used several lane names over time. Early notes mention alpha, beta, concierge, and recovery lanes, but those names were retired. Do not implement retired names unless a later rule explicitly reinstates them.

Section 02 - dashboard terminology. The old dashboard grouped cases by owner team, not by execution lane. Dashboard terms such as blue queue, amber queue, north queue, and west queue were display-only terms and must not be returned by route_case.

Section 03 - event severity notes. Severity values are written as p0, p1, p2, p3, and p4. Treat case differences as irrelevant. Some old incident tickets used sev1 or sev2, but the current routing table only relies on p0 and p1 for one mobile rule.

Section 04 - customer tiers. Tiers include standard, growth, enterprise, internal, and partner. Enterprise status matters only for the architecture migration rule. Growth does not imply enterprise.

Section 05 - billing background. Many billing cases are routine and should stay in the default lane. A refund request alone is not enough to escalate unless the refund amount is high or the case has a chargeback signal.

Section 06 - security background. The security group asked for fast containment on credential exposure and on incident codes using the SEC prefix. This is a current rule, not background. The exact current lane name appears in the policy table below.

Section 07 - privacy background. General privacy labels are not enough for escalation because many account tickets mention privacy in free text. The current rule requires both an EU region and a true data_subject_request flag.

Section 08 - mobile background. Mobile stability incidents are noisy. The current fast lane applies only when the component is mobile, severity is p0 or p1, and the symptom mentions crash. Slow login, UI copy, or install confusion remains standard.

Section 09 - architecture background. Migration and schema tickets caused prior outages for enterprise customers. The current rule is intentionally narrow: it applies to enterprise tickets involving the migration component or a schema label.

Section 10 - retired policy. The legacy route named migration_swat was retired. Do not return migration_swat. The replacement is in the current policy table.

Section 11 - retired policy. The old route named fraud_review was merged into billing triage. Do not return fraud_review for refunds. The replacement route for high-value refund or chargeback cases is in the current table.

Section 12 - retired policy. The old route named emergency_mobile was renamed. Do not return emergency_mobile.

Section 13 - retired policy. The old route named gdpr_triage was renamed. Do not return gdpr_triage.

Section 14 - notes on precedence. If multiple rules match, choose the first matching rule in the current policy table. This avoids sending credential leaks to lower-priority business lanes.

Section 15 - current policy table begins. Implement exactly these return values in route_case(ticket). Normalize strings case-insensitively. Missing fields should be treated as empty strings, empty lists, false booleans, or zero numeric values.

Rule 1: If incident_code starts with SEC- OR labels contains credential_leak, return containment.

Rule 2: Else if customer_tier is enterprise AND (component is migration OR labels contains schema), return principal_architect.

Rule 3: Else if labels contains chargeback OR amount_usd is greater than or equal to 1000 and labels contains refund, return revenue_guardian.

Rule 4: Else if component is mobile AND severity is p0 or p1 AND symptom contains crash, return mobile_hotfix.

Rule 5: Else if region is eu AND data_subject_request is true, return privacy_review.

Rule 6: Else return standard.

Section 16 - current policy table ends. The exact lane strings are containment, principal_architect, revenue_guardian, mobile_hotfix, privacy_review, and standard.

Section 17 - examples that are not tests. A low-value refund should stay standard. A privacy label outside EU should stay standard. An enterprise documentation question without migration or schema should stay standard.

Section 18 - examples that are current. SEC-441 with any tier is containment. A partner ticket with credential_leak is containment. Enterprise schema planning is principal_architect. Growth tier schema planning is not principal_architect unless another rule matches.

Section 19 - more context. The policy was designed to minimize unnecessary specialist escalation. Do not add logging, persistence, external calls, or imports beyond the Python standard library.

Section 20 - data shape. ticket is a plain dict. labels is normally a list of strings, but defensive code may tolerate missing labels or a single string label. The benchmark only requires normal dict inputs.

Section 21 - output shape. route_case must return a string. Do not return objects, enums, tuples, explanations, or confidence scores.

Section 22 - implementation boundary. The route_case implementation should be deterministic and side-effect free.

Section 23 - project note. This long-context benchmark is intentionally independent of hippo_memory internals so that it isolates memory retrieval from project architecture knowledge.

Section 24 - old exception. A 2024 note said that p0 enterprise mobile incidents should go to principal_architect. That exception was deleted and must not be implemented.

Section 25 - old exception. A 2024 note said chargeback in the EU should go to privacy_review. That exception was deleted. The current precedence table still sends chargeback to revenue_guardian unless a higher rule matches.

Section 26 - old exception. A 2024 note said all p0 cases go to containment. That was too broad. The current containment rule is limited to SEC incident codes or credential_leak labels.

Section 27 - audit note. The current policy table was approved because it is compact enough to fit in a memory pack while the surrounding archive is not compact.

Section 28 - final warning. The prompt for the coding task will not include these rules. A baseline model without memory should not be expected to know them.
