from app import build_engine, TickerThread, make_app
# Entrypoint
if __name__ == "__main__":
    eng = build_engine()
    tick = TickerThread(eng, key="capture", fps=5.0)
    tick.start()
    # Run detection every 4 seconds
    tick_detect = TickerThread(eng, key="detect", fps=0.25)  # every 4 seconds
    tick_detect.start()
    app = make_app(eng)
    # Run Flask
    app.run(host="0.0.0.0", port=5000, debug=False)
