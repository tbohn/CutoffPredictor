# CutoffPredictor
Tool for water utilities to monitor and predict customers' risk of service interruption

## Inputs
1. Database containing the following [tables](Documentation/database_tables.md):
 - meter
   - information about the meters, including size, customer type, and location_id
 - meter_location
   - information about meter locations (address, lat/lon, municipality)
 - occupant
   - information about occupants (customers) including location, move in/out dates
 - volume
   - records of water consumption, including meter ids, meter read dates and volume consumed
 - charge
   - records of billing, including location/occupant ids, billing dates, total consumption charges, late charges, etc
 - cutoffs
   - records of cutoffs, including location/occupant ids and cutoff dates

2. Configuration file:
 - contains information about:
   - paths to locations of code and data
   - runtime options (which stage of processing to run)
   - database access parameters
   - mapping access parameters
   - model types and feature engineering parameters
   - date on which to make the prediction



