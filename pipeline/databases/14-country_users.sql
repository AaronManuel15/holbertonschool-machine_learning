-- creates a table 'users' 
CREATE TABLE if NOT EXISTS users (
	id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
	email VARCHAR(255) NOT NULL UNIQUE,
	name VARCHAR(255),
	country ENUM('US', 'CO', 'TN') default 'US',
);
