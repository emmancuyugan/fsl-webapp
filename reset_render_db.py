from api import app, db

if __name__ == "__main__":
    print("Resetting database... This will delete all data.")
    with app.app_context():
        db.drop_all()
        db.create_all()
    print("Database has been reset successfully!")
