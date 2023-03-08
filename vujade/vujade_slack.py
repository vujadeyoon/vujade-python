import os
import re
import traceback
import slack_sdk
import slack_sdk.errors
from vujade import vujade_time as time_
from vujade.vujade_debug import printd


class DEBUG(object):
    def __init__(self):
        super(DEBUG, self).__init__()
        self.fileName = None
        self.lineNumber = None
        self.reTraceStack = re.compile('File \"(.+?)\", line (\d+?), .+')

    def get_file_line(self):
        for line in traceback.format_stack()[::-1]:
            m = re.match(self.reTraceStack, line.strip())
            if m:
                fileName = m.groups()[0]

                # ignore case
                if fileName == __file__:
                    continue
                self.fileName = os.path.split(fileName)[1]
                self.lineNumber = m.groups()[1]

                return True

        return False


class Slack(object):
    def __init__(self, _token_user: str, _token_bot: str, _channel: str, _is_time: bool= True, _is_debug: bool = False) -> None:
        super(Slack, self).__init__()
        """
        Required scopes of the OAuth & Permissions.
            i) Bot Token Scopes:
                - chat:write
            ii) User Token Scopes:
                - channels:history
                - channels:read
                - chat:write
                - files:read
                - files:write
                - groups:history
                - groups:read
                - im:history
                - im:read
                - mpim:history
                - mpim:read
                - users:read
        """
        self.token_user = _token_user
        self.token_bot = _token_bot
        self.channel = _channel
        self.is_time = _is_time
        self.is_debug = _is_debug
        self.client_bot = slack_sdk.WebClient(token=self.token_user)
        self.name_class = self.__class__.__name__

    def post_msg(self, *_args) -> None:
        info_str = ''
        for _idx, _arg in enumerate(_args):
            info_str += '{} '.format(_arg)
        info_str = info_str.rstrip(' ')

        if self.is_time is True:
            debug_time = time_.get_datetime()['readable'][:-4]
            info_trace = '[{}]'.format(debug_time)
        else:
            info_trace = ''

        if self.is_debug is True:
            debug_info = DEBUG()
            debug_info.get_file_line()
            info_trace = self._add_msg(_info_trace=info_trace, _add_msg='[{}: {}]'.format(debug_info.fileName, debug_info.lineNumber))

        info_trace = self._add_msg(_info_trace=info_trace, _add_msg=(info_str + '\n'))

        try:
            response = self.client_bot.chat_postMessage(channel=self.channel, text=info_trace)
            if response.status_code >= 500:
                printd('It is failed to send messages via {}'.format(self.name_class), _is_pause=False)
        except slack_sdk.errors.SlackApiError as e:
            printd('It is failed to send messages via {} with the Exception: {}.'.format(self.name_class, e.response['error']), _is_pause=False)

    def _add_msg(self, _info_trace: str, _add_msg: str) -> str:
        if len(_info_trace) != 0:
            _info_trace += ' '

        _info_trace += _add_msg

        return _info_trace


if __name__=='__main__':
    token_user = 'secret'
    token_bot = 'secret'

    slack = Slack(_token_user=token_user, _token_bot=token_bot, _channel='#random')
    slack.post_msg('message')
