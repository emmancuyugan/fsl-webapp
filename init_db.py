from app import app, db

# Create database tables
with app.app_context():
    db.create_all()
    print("Database tables created successfully!")

    # Test the connection by querying the User table
    try:
        result = db.session.execute(db.text('SELECT 1')).first()
        print("Database connection successful!")
    except Exception as e:
        print(f"Database connection failed: {e}")
