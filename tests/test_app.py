"""Interaction checks for the Streamlit interface; run from the project folder."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class InterfaceTests(unittest.TestCase):
    def test_conversation_controls(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=45).run()
        self.assertFalse(app.exception)

        app.button(key='ask_Reset password').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.messages), 2)
        self.assertIn('password', app.session_state.messages[-1]['result']['text'])

        app.selectbox(key='decoder').select('Beam search (3)').run()
        app.chat_input[0].set_value('hi').run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.messages), 4)
        self.assertEqual(app.session_state.messages[-1]['result']['decoding'], 'Beam search (3)')

        app.toggle(key='attention').set_value(False).run()
        app.checkbox(key='only_misses').check().run()
        self.assertFalse(app.exception)

        app.chat_input[0].set_value('😊').run()
        self.assertFalse(app.exception)
        self.assertIn('Please enter', app.session_state.messages[-1]['result']['text'])

        app.chat_input[0].set_value('how do i reset my password ' * 5).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state.messages[-1]['result']['truncated'])

        app.button(key='new_chat').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.messages), 0)


if __name__ == '__main__':
    unittest.main()
