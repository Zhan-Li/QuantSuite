import schedule


def job():
    schedule.every().day.at("06:00").do(job)
    schedule.every().day.at("12:00").do(job)
    schedule.every().day.at("18:00").do(job)
    schedule.every().day.at("00:00").do(job)


while True:
    schedule.run_pending()
