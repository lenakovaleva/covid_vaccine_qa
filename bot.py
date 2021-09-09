import logging
import main
from main import ls_questions, ls_answers, cos_sim, cos_sim_sorted, user_sent_prep
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        text='Добрый день! \n\nВас приветствует бот по вопросам вакцинации. \n\nЗадавайте мне ваши вопросы, а я дам ответ в соответствии с последними рекомендациями ВОЗ. \n\nМожете использовать команды /start, /help и /search.',
        reply_markup=markup)
    return 1

reply_keyboard = [['/start', '/help'],
                      ['/search']]

markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=False)

def help(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Команда /start показывает приветственное сообщение. \n\nКоманда /search подскажет как правильно искать ответ.')

def search(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Можно написать вопрос полностью, а можно указать только ключевые слова. \n\nПоиск только на русском языке.')

def stop(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Нажмите /start чтобы увидеть первое сообщение и начать поиск')
    return ConversationHandler.END

def print_qstn_and_answ(update: Update, context: CallbackContext):
    update.message.reply_text(
        text='Вопрос:',
    )
    found_sent = find_sentence(update.message.text)

    update.message.reply_text(
        text = found_sent
    )
    update.message.reply_text(
        text='Ответ:',
    )

    found_answ = find_answer(update.message.text)
    update.message.reply_text(
        text=found_answ
    )

def find_sentence(input_question):
    text = ls_questions[cos_sim(user_sent_prep(input_question))]
    return text

def find_answer(input_question):
    text = ls_answers[cos_sim(user_sent_prep(input_question))]
    return text

def main():
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            1: [CommandHandler('start', start), CommandHandler('help', help), CommandHandler('search', search), CommandHandler('stop', stop), MessageHandler(Filters.text, print_qstn_and_answ, pass_user_data=True)]
        },

        fallbacks=[CommandHandler('stop', stop)]
    )

    print('----  *****  Bot has started working   *****  -----')

    updater = Updater(
        token='',
        use_context=True,
    )
    dp = updater.dispatcher

    dp.add_handler(conv_handler)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()