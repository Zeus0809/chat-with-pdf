import flet as ft
from flet.core.control import OptionalNumber


submenu_button = ft.ButtonStyle(
    padding=10
)

menubar = ft.MenuStyle(
    
)

class ChatStyles:
    """Chat-specific styling definitions."""

    COLORS = {
        "user_bubble": ft.Colors.BLUE_400,
        "agent_bubble": ft.Colors.WHITE,
        "text": ft.Colors.BLACK87,
        "background": ft.Colors.GREY_100,
        "border": ft.Colors.GREY_300
    }

    MAX_BUBBLE_WIDTH_RATIO = 0.8 # 4/5 of the chat area

    BASE_BUBBLE = {
        "border_radius": ft.border_radius.all(12),
        "padding": ft.padding.all(12),
        "margin": ft.margin.only(bottom=8)
    }

    @staticmethod
    def user_bubble() -> dict:
        """Returns styling for user message bubbles."""
        return {
            **ChatStyles.BASE_BUBBLE,
            "bgcolor": ChatStyles.COLORS['user_bubble']
        }
    
    @staticmethod
    def agent_bubble() -> dict:
        """Returns styling for agent message bubbles."""
        return {
            **ChatStyles.BASE_BUBBLE,
            "bgcolor": ChatStyles.COLORS['agent_bubble'],
            "border": ft.border.all(1, ChatStyles.COLORS['border'])
        }
    
    @staticmethod
    def create_user_message_row(bubble_content: ft.Text, parent_width: OptionalNumber) -> ft.Row:
        """Build and return a styled and aligned row that contains a user message bubble, intended to be placed in the chat area."""

        assert isinstance(bubble_content, ft.Text), f"bubble_content should be a Flet text element, instead got: {type(bubble_content)}"
        assert isinstance(parent_width, (int, float)), f"parent_width should be an int or a float, instead got: {type(parent_width)}"

        max_bubble_width = parent_width * ChatStyles.MAX_BUBBLE_WIDTH_RATIO

        bubble_content.selectable = True
        bubble_content.no_wrap = False

        message_bubble = ft.Column( # using a Column with width and alignment solved the wrapping issue!
            controls=[
                ft.Container(
                    content=bubble_content,
                    **ChatStyles.user_bubble()
                )
            ],
            width=max_bubble_width, # this is super important
            horizontal_alignment=ft.CrossAxisAlignment.END # this is super important
        )

        return ft.Row(
            controls=[message_bubble],
            alignment=ft.MainAxisAlignment.END
        )
    
    @staticmethod
    def create_agent_message_row(bubble_content: ft.Text, parent_width: OptionalNumber) -> ft.Row:
        """Build and return a styled and aligned row that contains an agent message bubble, intended to be placed in the chat area."""
        
        assert isinstance(bubble_content, ft.Text), f"bubble_content should be a Flet text element, instead got: {type(bubble_content)}"
        assert isinstance(parent_width, (int, float)), f"parent_width should be an int or a float, instead got: {type(parent_width)}"
        
        max_bubble_width = parent_width * ChatStyles.MAX_BUBBLE_WIDTH_RATIO

        bubble_content.selectable = True
        bubble_content.no_wrap = False

        message_bubble = ft.Column(
            controls=[
                ft.Container(
                    content=bubble_content,
                    **ChatStyles.agent_bubble()
                )
            ],
            width=max_bubble_width,
            horizontal_alignment=ft.CrossAxisAlignment.START
        )

        return ft.Row(
            controls=[message_bubble],
            alignment=ft.MainAxisAlignment.START
        )

    
class TextStyles:
    """Text styling definitions."""

    @staticmethod
    def message_text():
        """Unified text style for both user and agent messages."""
        return {
            "color": ChatStyles.COLORS['text'],
            "size": 14
        }



