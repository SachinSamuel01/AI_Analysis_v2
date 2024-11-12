import pandas as pd

path= "please provide path to the designated csv file "
df= pd.read_csv(path)

'''
Easy:

Which airline has the most flights listed?
What are the top three most frequented destinations?
Number of bookings for American Airlines yesterday.

Medium:

Average flight delay per airline.
Month with the highest number of bookings.

Hard:

Patterns in booking cancellations, focusing on specific days or airlines with high cancellation rates.
Analyze seat occupancy to find the most and least popular flights.

'''




'''
['airline_id', 'airline_name', 'flight_number', 'scheduled_departure',
       'scheduled_arrival', 'departure_time_local', 'arrival_time_local',
       'booking_code', 'passenger_name', 'seat_number', 'travel_class',
       'ticket_fare', 'additional_services', 'loyalty_points',
       'booking_status', 'gate', 'terminal', 'baggage_claim',
       'flight_duration_hours', 'number_of_layovers', 'layover_locations',
       'aircraft_type', 'pilot_name', 'cabin_crew', 'inflight_entertainment',
       'meal_option', 'wifi_availability', 'window_seat', 'aisle_seat',
       'emergency_exit_row', 'number_of_stops', 'loyalty_program_member']

'''

with open("test_cases.txt",'a') as f:

    # f.write(' ')
    # exit()
# Which airline has the most flights listed?
    airline_fight= df.groupby('airline_name').flight_number.count().reset_index()
    airline_flights = airline_fight.sort_values('flight_number', ascending=False)
    top_airline= airline_flights.iloc[0]
    f.write("Which airline has the most flights listed?")
    f.write(f"The airline with the most flights is {top_airline['Airline']} with {top_airline['Number of Flights']} flights")
    f.write()


    # What are the top three most frequented destinations?
    top_destinations = (df['layover_locations']
                    .value_counts()
                    .head(3))
    f.write('What are the top three most frequented destinations?')
    f.write('\n')
    f.write(f"{top_destinations}")
    f.write('\n')
    f.write("-----------------------------------------------------------------------------------------")
    f.write('\n')
    f.write('\n')


    # Number of bookings for American Airlines yesterday.
    bookings = df[
        (df['airline_name'] == 'American Airlines') & 
        (pd.to_datetime(df['scheduled_departure']).dt.date == (pd.Timestamp.today().date() - pd.Timedelta(days=1)))
    ].shape[0]
    f.write("Number of bookings for American Airlines yesterday.")
    f.write('\n')
    f.write(f"{bookings}")
    f.write('\n')
    
    f.write("-----------------------------------------------------------------------------------------")
    f.write('\n')
    f.write('\n')


    # Average flight delay per airline.
    df['scheduled_time'] = pd.to_datetime(df['scheduled_departure']).dt.time
    df['actual_time'] = pd.to_datetime(df['departure_time_local']).dt.time

    # Convert times to minutes since midnight for calculation
    def time_to_minutes(t):
        return t.hour * 60 + t.minute

    df['scheduled_minutes'] = df['scheduled_time'].apply(time_to_minutes)
    df['actual_minutes'] = df['actual_time'].apply(time_to_minutes)

    # Calculate delay in minutes
    df['delay_minutes'] = df['actual_minutes'] - df['scheduled_minutes']

    # Handle cases where flight departed after midnight
    # df.loc[df['delay_minutes'] < -720, 'delay_minutes'] += 1440  # Add 24 hours in minutes

    # Calculate average delay per airline
    average_delay = df.groupby(['airline_id', 'airline_name'])['delay_minutes'].mean().reset_index()

    f.write('Average flight delay per airline.')
    f.write('\n')
    f.write(f"{average_delay}")
    f.write('\n')
    
    f.write("-----------------------------------------------------------------------------------------")
    f.write('\n')
    f.write('\n')



    # Month with the highest number of bookings.
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])

    # Extract month and count bookings
    monthly_bookings = df['scheduled_departure'].dt.month.value_counts()

    # Get month with highest bookings
    highest_month = monthly_bookings.index[0]

    # If you want month name instead of number
    highest_month_name = pd.Timestamp(2024, highest_month, 1).strftime('%B')

    f.write("Month with the highest number of bookings.")
    f.write('\n')
    f.write(f"{highest_month_name}")
    f.write('\n')
    f.write("-----------------------------------------------------------------------------------------")
    f.write('\n')
    f.write('\n')


    # Patterns in booking cancellations, focusing on specific days or airlines with high cancellation rates.
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])

    # Extract day of week from scheduled_departure
    df['day_of_week'] = df['scheduled_departure'].dt.day_name()

    # Analyze cancellation patterns by day of week
    day_cancellations = df[df['booking_status'] == 'Cancelled'].groupby('day_of_week').size()
    day_total = df.groupby('day_of_week').size()
    day_cancellation_rate = (day_cancellations / day_total * 100).round(2)

    # Analyze cancellation patterns by airline
    airline_cancellations = df[df['booking_status'] == 'Cancelled'].groupby('airline_name').size()
    airline_total = df.groupby('airline_name').size() 
    airline_cancellation_rate = (airline_cancellations / airline_total * 100).round(2)

    # Get top 5 airlines with highest cancellation rates
    top_airlines_cancellations = airline_cancellation_rate.nlargest(5)

    # Get days with highest cancellation rates
    top_days_cancellations = day_cancellation_rate.nlargest(3)

    f.write("Patterns in booking cancellations, focusing on specific days or airlines with high cancellation rates.")
    f.write('\n')
    f.write("Top 5 Airlines with Highest Cancellation Rates:")
    f.write('\n')
    f.write(f"{top_airlines_cancellations}")
    f.write('\n')
    f.write("Days with Highest Cancellation Rates:")
    f.write('\n')
    f.write(f"{top_days_cancellations}")
    f.write('\n')
    f.write("-----------------------------------------------------------------------------------------")
    f.write('\n')
    f.write('\n')



    # Analyze seat occupancy to find the most and least popular flights.

    # Group by flight details and count passengers
    seat_analysis = (df.groupby(['airline_name', 'flight_number', 'scheduled_departure'])
                    .agg({
                        'passenger_name': 'count',  # Count passengers per flight
                        'seat_number': 'nunique'    # Count unique seats
                    })
                    .reset_index()
                    .rename(columns={'passenger_name': 'passenger_count'})
    )

    # Find most popular flights
    most_popular = (seat_analysis
                    .sort_values('passenger_count', ascending=False)
                    .head(5))

    # Find least popular flights
    least_popular = (seat_analysis
                    .sort_values('passenger_count', ascending=True)
                    .head(5))

    # Calculate occupancy rate if we assume total seats = max seats observed for that flight
    seat_analysis['occupancy_rate'] = seat_analysis['passenger_count'] / seat_analysis['seat_number'] * 100

    # Get flights by occupancy rate
    flights_by_occupancy = (seat_analysis
                        .sort_values('occupancy_rate', ascending=False)
                        .round({'occupancy_rate': 2}))

    # Display results
    f.write("Analyze seat occupancy to find the most and least popular flights.")
    f.write('\n')
    f.write("Most Popular Flights by Passenger Count:")
    f.write('\n')
    f.write(f"{most_popular}")
    f.write('\n')
    f.write("\nLeast Popular Flights by Passenger Count:")
    f.write('\n')
    f.write(f"{least_popular}")
    f.write('\n')
    f.write("\nFlights by Occupancy Rate:")
    f.write('\n')
    f.write(f"{flights_by_occupancy}")
    f.write('\n')
    f.write('\n')