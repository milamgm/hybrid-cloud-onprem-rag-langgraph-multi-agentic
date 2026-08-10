package onyx.tools

import rego.v1

default decision := {
    "allow": false,
    "requires_approval": false,
    "reason": "Tool call is not permitted by policy.",
}

decision := {
    "allow": true,
    "requires_approval": false,
    "reason": "Corpus search is permitted.",
} if {
    input.tool_name == "search_corpus"
    valid_query
    limit_within(5)
}

decision := {
    "allow": true,
    "requires_approval": false,
    "reason": "Approved web search is permitted.",
} if {
    input.tool_name == "search_web"
    "tool.web.search" in input.roles
    input.data_classification in {"public", "internal"}
    valid_query
    limit_within(5)
}

decision := {
    "allow": true,
    "requires_approval": true,
    "reason": "A recorded human approval is required for this sensitive tool.",
} if {
    input.tool_name in {"send_email", "write_record", "delete_record"}
    "tool.sensitive.execute" in input.roles
    input.approval_id == null
}

decision := {
    "allow": true,
    "requires_approval": true,
    "reason": "Sensitive tool is permitted with recorded human approval.",
} if {
    input.tool_name in {"send_email", "write_record", "delete_record"}
    "tool.sensitive.execute" in input.roles
    is_string(input.approval_id)
    count(input.approval_id) > 0
}

valid_query if {
    query := input.arguments.query
    is_string(query)
    count(query) > 0
    count(query) <= 1000
}

limit_within(maximum) if {
    input.requested_limit == null
}

limit_within(maximum) if {
    is_number(input.requested_limit)
    input.requested_limit > 0
    input.requested_limit <= maximum
}
