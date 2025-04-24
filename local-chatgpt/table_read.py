import sqlite3

# Connect to the database file
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

# Run a SELECT query
cursor.execute("SELECT program_name, enrolled, capacity FROM enrollment")
rows = cursor.fetchall()

# Print the results
for row in rows:
    print(f"Program Name: {row[0]}, Enrolled: {row[1]}, Capacity: {row[2]}")

# Clean up
cursor.close()
conn.close()
