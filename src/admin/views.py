"""SQLAdmin model views for admin panel."""

from sqladmin import ModelView

from src.data.models import (
    AuditLog,
    Dialog,
    Message,
    Model,
    SystemConfig,
    TokenBalance,
    TokenTransaction,
)


class ModelAdmin(ModelView, model=Model):
    """Admin view for LLM Models."""

    name = "Model"
    name_plural = "Models"
    icon = "fa-solid fa-robot"

    column_list = [
        Model.name,
        Model.provider,
        Model.cost_per_1k_prompt_tokens,
        Model.cost_per_1k_completion_tokens,
        Model.context_window,
        Model.enabled,
        Model.updated_at,
    ]
    column_searchable_list = [Model.name, Model.provider]
    column_sortable_list = [
        Model.name,
        Model.provider,
        Model.enabled,
        Model.cost_per_1k_prompt_tokens,
        Model.updated_at,
    ]
    column_default_sort = ("name", False)

    form_columns = [
        Model.name,
        Model.provider,
        Model.cost_per_1k_prompt_tokens,
        Model.cost_per_1k_completion_tokens,
        Model.context_window,
        Model.enabled,
    ]

    column_labels = {
        Model.name: "Model Name",
        Model.provider: "Provider",
        Model.cost_per_1k_prompt_tokens: "Cost/1K Prompt",
        Model.cost_per_1k_completion_tokens: "Cost/1K Completion",
        Model.context_window: "Context Window",
        Model.enabled: "Enabled",
        Model.updated_at: "Updated At",
    }

    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True


class TokenBalanceAdmin(ModelView, model=TokenBalance):
    """Admin view for Token Balances."""

    name = "Token Balance"
    name_plural = "Token Balances"
    icon = "fa-solid fa-coins"

    column_list = [
        TokenBalance.user_id,
        TokenBalance.balance,
        TokenBalance.limit,
        TokenBalance.updated_at,
    ]
    column_searchable_list = [TokenBalance.user_id]
    column_sortable_list = [
        TokenBalance.user_id,
        TokenBalance.balance,
        TokenBalance.limit,
        TokenBalance.updated_at,
    ]
    column_default_sort = ("user_id", False)

    # Include user_id in form (it's a primary key but we need to set it manually)
    form_columns = [
        TokenBalance.user_id,
        TokenBalance.balance,
        TokenBalance.limit,
    ]
    form_include_pk = True  # Show primary key in create/edit forms

    column_labels = {
        TokenBalance.user_id: "User ID",
        TokenBalance.balance: "Balance",
        TokenBalance.limit: "Limit (null=unlimited)",
        TokenBalance.updated_at: "Updated At",
    }

    can_create = True
    can_edit = True
    can_delete = False  # Don't allow deleting balances
    can_view_details = True


class TokenTransactionAdmin(ModelView, model=TokenTransaction):
    """Admin view for Token Transactions (read-only audit log)."""

    name = "Token Transaction"
    name_plural = "Token Transactions"
    icon = "fa-solid fa-exchange-alt"

    column_list = [
        TokenTransaction.id,
        TokenTransaction.user_id,
        TokenTransaction.amount,
        TokenTransaction.reason,
        TokenTransaction.admin_user_id,
        TokenTransaction.created_at,
    ]
    column_searchable_list = [TokenTransaction.user_id, TokenTransaction.reason]
    column_sortable_list = [
        TokenTransaction.id,
        TokenTransaction.user_id,
        TokenTransaction.amount,
        TokenTransaction.reason,
        TokenTransaction.created_at,
    ]
    column_default_sort = ("created_at", True)  # Newest first

    column_labels = {
        TokenTransaction.id: "ID",
        TokenTransaction.user_id: "User ID",
        TokenTransaction.amount: "Amount",
        TokenTransaction.reason: "Reason",
        TokenTransaction.admin_user_id: "Admin ID",
        TokenTransaction.created_at: "Created At",
    }

    can_create = False  # Read-only
    can_edit = False
    can_delete = False
    can_view_details = True


class DialogAdmin(ModelView, model=Dialog):
    """Admin view for Dialogs."""

    name = "Dialog"
    name_plural = "Dialogs"
    icon = "fa-solid fa-comments"

    column_list = [
        Dialog.id,
        Dialog.user_id,
        Dialog.title,
        Dialog.model_name,
        Dialog.created_at,
        Dialog.updated_at,
    ]
    column_searchable_list = [Dialog.title, Dialog.model_name]
    column_sortable_list = [
        Dialog.user_id,
        Dialog.title,
        Dialog.model_name,
        Dialog.created_at,
        Dialog.updated_at,
    ]
    column_default_sort = ("created_at", True)

    column_labels = {
        Dialog.id: "ID",
        Dialog.user_id: "User ID",
        Dialog.title: "Title",
        Dialog.model_name: "Model",
        Dialog.created_at: "Created At",
        Dialog.updated_at: "Updated At",
    }

    can_create = False  # Created via API
    can_edit = True
    can_delete = True
    can_view_details = True


class MessageAdmin(ModelView, model=Message):
    """Admin view for Messages."""

    name = "Message"
    name_plural = "Messages"
    icon = "fa-solid fa-message"

    column_list = [
        Message.id,
        Message.dialog_id,
        Message.role,
        Message.prompt_tokens,
        Message.completion_tokens,
        Message.created_at,
    ]
    column_searchable_list = [Message.role]
    column_sortable_list = [
        Message.dialog_id,
        Message.role,
        Message.created_at,
    ]
    column_default_sort = ("created_at", True)

    # Truncate content in list view
    column_formatters = {
        Message.content: lambda m, a: (
            m.content[:100] + "..." if len(m.content) > 100 else m.content
        ),
    }

    column_labels = {
        Message.id: "ID",
        Message.dialog_id: "Dialog ID",
        Message.role: "Role",
        Message.content: "Content",
        Message.prompt_tokens: "Prompt Tokens",
        Message.completion_tokens: "Completion Tokens",
        Message.created_at: "Created At",
    }

    can_create = False  # Created via API
    can_edit = False
    can_delete = True
    can_view_details = True


class AuditLogAdmin(ModelView, model=AuditLog):
    """Admin view for Audit Logs (read-only)."""

    name = "Audit Log"
    name_plural = "Audit Logs"
    icon = "fa-solid fa-clipboard-list"

    column_list = [
        AuditLog.id,
        AuditLog.user_id,
        AuditLog.action,
        AuditLog.resource_type,
        AuditLog.resource_id,
        AuditLog.ip_address,
        AuditLog.created_at,
    ]
    column_searchable_list = [AuditLog.action, AuditLog.resource_type, AuditLog.resource_id]
    column_sortable_list = [
        AuditLog.id,
        AuditLog.user_id,
        AuditLog.action,
        AuditLog.resource_type,
        AuditLog.created_at,
    ]
    column_default_sort = ("created_at", True)

    column_labels = {
        AuditLog.id: "ID",
        AuditLog.user_id: "User ID",
        AuditLog.action: "Action",
        AuditLog.resource_type: "Resource Type",
        AuditLog.resource_id: "Resource ID",
        AuditLog.ip_address: "IP Address",
        AuditLog.created_at: "Created At",
    }

    can_create = False  # Read-only
    can_edit = False
    can_delete = False
    can_view_details = True


class SystemConfigAdmin(ModelView, model=SystemConfig):
    """Admin view for System Configuration (API keys, settings)."""

    name = "System Config"
    name_plural = "System Configs"
    icon = "fa-solid fa-gear"

    column_list = [
        SystemConfig.key,
        SystemConfig.is_secret,
        SystemConfig.description,
        SystemConfig.updated_at,
        SystemConfig.updated_by,
    ]
    column_searchable_list = [SystemConfig.key, SystemConfig.description]
    column_sortable_list = [
        SystemConfig.key,
        SystemConfig.is_secret,
        SystemConfig.updated_at,
    ]
    column_default_sort = ("key", False)

    form_columns = [
        SystemConfig.key,
        SystemConfig.value,
        SystemConfig.is_secret,
        SystemConfig.description,
    ]

    # Hide secret values in list view
    column_formatters = {
        SystemConfig.value: lambda m, a: "********" if m.is_secret else m.value,
    }

    column_labels = {
        SystemConfig.key: "Key",
        SystemConfig.value: "Value",
        SystemConfig.is_secret: "Secret",
        SystemConfig.description: "Description",
        SystemConfig.updated_at: "Updated At",
        SystemConfig.updated_by: "Updated By",
    }

    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True
